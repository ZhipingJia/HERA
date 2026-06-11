from __future__ import  absolute_import
import os
from collections import namedtuple
import time
from torch.nn import functional as F
from model.utils.creator_tool import AnchorTargetCreator, ProposalTargetCreator

import sys
sys.path.append("../model_adapter")
from quantization_and_noise.quant_layer import QuanModule
from utils.config import get_count_loss

from torch import nn
import torch as t
from utils import array_tool as at

from utils.config import opt
from utils.meters import ConfusionMeter, AverageValueMeter
import numpy as np
from collections import OrderedDict

LossTuple = namedtuple('LossTuple',
                       ['rpn_loc_loss',
                        'rpn_cls_loss',
                        'roi_loc_loss',
                        'roi_cls_loss',
                        'total_loss'
                        ])


class FasterRCNNTrainer(nn.Module):
    """wrapper for conveniently training. return losses

    The losses include:

    * :obj:`rpn_loc_loss`: The localization loss for \
        Region Proposal Network (RPN).
    * :obj:`rpn_cls_loss`: The classification loss for RPN.
    * :obj:`roi_loc_loss`: The localization loss for the head module.
    * :obj:`roi_cls_loss`: The classification loss for the head module.
    * :obj:`total_loss`: The sum of 4 loss above.

    Args:
        faster_rcnn (model.FasterRCNN):
            A Faster R-CNN model that is going to be trained.
    """

    def __init__(self, faster_rcnn):
        super(FasterRCNNTrainer, self).__init__()

        self.faster_rcnn = faster_rcnn
        self.rpn_sigma = opt.rpn_sigma
        self.roi_sigma = opt.roi_sigma

        # target creator create gt_bbox gt_label etc as training targets. 
        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()

        self.loc_normalize_mean = faster_rcnn.loc_normalize_mean
        self.loc_normalize_std = faster_rcnn.loc_normalize_std

        self.optimizer = self.faster_rcnn.get_optimizer()
        # visdom wrapper
        # self.vis = Visualizer(env=opt.env)

        # indicators for training status
        self.rpn_cm = ConfusionMeter(2)
        self.roi_cm = ConfusionMeter(2)
        self.meters = {k: AverageValueMeter() for k in LossTuple._fields}  # average loss
        self.loss_layer = opt.weight_count_loss

        self.loss_layer = opt.weight_count_loss

    def forward(self, imgs, bboxes, labels, scale):
        """Forward Faster R-CNN and calculate losses.

        Here are notations used.

        * :math:`N` is the batch size.
        * :math:`R` is the number of bounding boxes per image.

        Currently, only :math:`N=1` is supported.

        Args:
            imgs (~torch.autograd.Variable): A variable with a batch of images.
            bboxes (~torch.autograd.Variable): A batch of bounding boxes.
                Its shape is :math:`(N, R, 4)`.
            labels (~torch.autograd..Variable): A batch of labels.
                Its shape is :math:`(N, R)`. The background is excluded from
                the definition, which means that the range of the value
                is :math:`[0, L - 1]`. :math:`L` is the number of foreground
                classes.
            scale (float): Amount of scaling applied to
                the raw image during preprocessing.

        Returns:
            namedtuple of 5 losses
        """
        # import pdb
        # pdb.set_trace()
        n = bboxes.shape[0]
        if n != 1:
            raise ValueError('Currently only batch size 1 is supported.')

        _, _, H, W = imgs.shape
        img_size = (H, W)

        features = self.faster_rcnn.extractor(imgs)
        # print('feature size',features[0].shape)
        if len(features)==2:
            features = features[0]*features[1]
            #pass
        

        rpn_locs, rpn_scores, rois, roi_indices, anchor = \
            self.faster_rcnn.rpn(features, img_size, scale)

        # Since batch size is one, convert variables to singular form
        bbox = bboxes[0]
        label = labels[0]
        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]
        roi = rois

        # Sample RoIs and forward
        # it's fine to break the computation graph of rois, 
        # consider them as constant input
        sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
            roi,
            at.tonumpy(bbox),
            at.tonumpy(label),
            self.loc_normalize_mean,
            self.loc_normalize_std)
        # NOTE it's all zero because now it only support for batch=1 now
        sample_roi_index = t.zeros(len(sample_roi))
        roi_cls_loc, roi_score = self.faster_rcnn.head(
            features,
            sample_roi,
            sample_roi_index)

        # ------------------ RPN losses -------------------#
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
            at.tonumpy(bbox),
            anchor,
            img_size)
        gt_rpn_label = at.totensor(gt_rpn_label).long()
        gt_rpn_loc = at.totensor(gt_rpn_loc)
        rpn_loc_loss = _fast_rcnn_loc_loss(
            rpn_loc,
            gt_rpn_loc,
            gt_rpn_label.data,
            self.rpn_sigma)
        
        # NOTE: default value of ignore_index is -100 ...
        rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label.cuda(), ignore_index=-1)
        _gt_rpn_label = gt_rpn_label[gt_rpn_label > -1]
        
        _rpn_score = at.tonumpy(rpn_score)[at.tonumpy(gt_rpn_label) > -1]
        self.rpn_cm.add(at.totensor(_rpn_score, False), _gt_rpn_label.data.long())

        # ------------------ ROI losses (fast rcnn loss) -------------------#
        n_sample = roi_cls_loc.shape[0]
        roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
        roi_loc = roi_cls_loc[t.arange(0, n_sample).long().cuda(), \
                              at.totensor(gt_roi_label).long()]
        gt_roi_label = at.totensor(gt_roi_label).long()
        gt_roi_loc = at.totensor(gt_roi_loc)

        roi_loc_loss = _fast_rcnn_loc_loss(
            roi_loc.contiguous(),
            gt_roi_loc,
            gt_roi_label.data,
            self.roi_sigma)

        roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label.cuda())
        
        # import pdb
        # pdb.set_trace()

        self.roi_cm.add(at.totensor(roi_score, False), gt_roi_label.data.long())

        count_loss_list = []
        for name,module in self.faster_rcnn.named_modules():
            if "extractor." in name and type(module) in QuanModule and name in self.loss_layer:
                count_loss_list.append(get_count_loss(module)*self.loss_layer[name])

        count_loss = sum(count_loss_list)

        losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
        #losses = losses + [sum(losses)]
        losses = losses + [sum(losses)+count_loss]

        return LossTuple(*losses)

    def train_step(self, imgs, bboxes, labels, scale):
        self.optimizer.zero_grad()
        losses = self.forward(imgs, bboxes, labels, scale)
        self.update_meters(losses)
        # if losses.total_loss.item() > 1.5:
        #     print('loss is nan and skip weight update')
        #     return losses
        losses.total_loss.backward()
        self.optimizer.step()
        # print('weights grad:')
        # for name, param in self.faster_rcnn.named_parameters():
        #     if param.requires_grad:
        #         if 'extractor.0.0' in name or 'extractor.0.2' in name or 'extractor.1.1' in name or 'extractor.1.3' in name:
        #             print(name, param.grad.max())
        return losses

    def save(self, save_optimizer=False, save_path=None, **kwargs):
        """serialize models include optimizer and other info
        return path where the model-file is stored.

        Args:
            save_optimizer (bool): whether save optimizer.state_dict().
            save_path (string): where to save model, if it's None, save_path
                is generate using time str and info from kwargs.
        
        Returns:
            save_path(str): the path to save models.
        """
        save_dict = dict()

        save_dict['model'] = self.faster_rcnn.state_dict()
        save_dict['config'] = opt._state_dict()
        save_dict['other_info'] = kwargs
        # save_dict['vis_info'] = self.vis.state_dict()

        if save_optimizer:
            save_dict['optimizer'] = self.optimizer.state_dict()

        # if save_path is None:
        timestr = time.strftime('%m%d%H%M')
        save_path = f"{save_path}/fasterrcnn_{timestr}"
        for k_, v_ in kwargs.items():
            save_path += '_%s' % v_
        save_path += '.pth'

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        t.save(save_dict, save_path)
        # self.vis.save([self.vis.env])
        return save_path

    def load(self, path, load_optimizer=True, parse_opt=False, ):
        # state_dict = t.load(path)
        state_dict = t.load(path, map_location=t.device('cpu'))
        print('state_dict:')
        print(state_dict.keys())
        print(state_dict['model'].keys())
        # for name,param in self.faster_rcnn.named_parameters():
        #     if name in state_dict['model'] and param.shape != state_dict['model'][name].shape:
        #         state_dict['model'].pop(name)
        #     else:
        #         param.requires_grad = False
        # for key in list(state_dict['model'].keys()):
        #     if 'layer4' in key and 'running' in key:
        #         state_dict['model'].pop(key)
        # new_state_dict = OrderedDict()
        if 'model' in state_dict:
            # 复制整个state_dict
            new_state_dict = state_dict.copy()
            '''
            # 创建一个新的model_state_dict来存储修改后的模型状态
            new_model_state_dict = OrderedDict()

            for k, v in state_dict['model'].items():
                if 'extractor.1.0' in k:
                    new_key = k.replace('extractor.1.0', 'extractor.1.1')
                elif 'extractor.1.2' in k:
                    new_key = k.replace('extractor.1.2', 'extractor.1.3')
                else:
                    new_key = k  # 如果键不需要改变，就保持原样
                new_model_state_dict[new_key] = v

            # 用新的model_state_dict替换原来的state_dict['model']
            new_state_dict['model'] = new_model_state_dict
            '''
            # 加载新的state_dict
            # model.load_state_dict(new_state_dict)
            print(new_state_dict['model'].keys())
            self.faster_rcnn.load_state_dict(new_state_dict['model'], strict=False)
        else:  # legacy way, for backward compatibility
            self.faster_rcnn.load_state_dict(state_dict,  strict=True)
            return self
        if parse_opt:
            print('config!!!')
            opt._parse(state_dict['config'])
        if 'optimizer' in state_dict and load_optimizer:
            print('optimizer!!!')
            self.optimizer.load_state_dict(state_dict['optimizer'])
        return self

    def update_meters(self, losses):
        loss_d = {k: at.scalar(v) for k, v in losses._asdict().items()}
        for key, meter in self.meters.items():
            meter.add(loss_d[key])

    def reset_meters(self):
        for key, meter in self.meters.items():
            meter.reset()
        self.roi_cm.reset()
        self.rpn_cm.reset()

    def get_meter_data(self):
        return {k: v.value()[0] for k, v in self.meters.items()}


def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()


def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    in_weight = t.zeros(gt_loc.shape).cuda()
    # Localization loss is calculated only for positive rois.
    # NOTE:  unlike origin implementation, 
    # we don't need inside_weight and outside_weight, they can calculate by gt_label
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    # Normalize by total number of negtive and positive rois.
    loc_loss /= ((gt_label >= 0).sum().float()) # ignore gt_label==-1 for rpn_loss
    return loc_loss
