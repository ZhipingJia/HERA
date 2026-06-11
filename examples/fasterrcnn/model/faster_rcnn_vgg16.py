from __future__ import  absolute_import
import torch as t
from torch import nn
from torchvision.models import vgg16
from torchvision.ops import RoIPool
import torch.nn.functional as F

from model.region_proposal_network import RegionProposalNetwork
from model.faster_rcnn import FasterRCNN
from utils import array_tool as at
from utils.config import opt


ADDBIAS = True
# def Vgg16_net(channels_list=[64, 96, 128, 256, 256], block_num=5):
#     add_bias = False
#     layer1=nn.Sequential(
#         nn.Conv2d(in_channels=3,out_channels=channels_list[0],kernel_size=3,stride=1,padding=1,bias=add_bias), #(32-3+2)/1+1=32   32*32*64
#         nn.ReLU(inplace=True),

#         nn.Conv2d(in_channels=channels_list[0],out_channels=channels_list[0],kernel_size=3,stride=1,padding=1,bias=add_bias), 
#         nn.ReLU(inplace=True),

#         #nn.MaxPool2d(kernel_size=2,stride=2)   #(32-2)/2+1=16         16*16*64
#         nn.AvgPool2d(kernel_size=2,stride=2)
#     )

#     layer2=nn.Sequential(
#         nn.Conv2d(in_channels=channels_list[0],out_channels=channels_list[1],kernel_size=3,stride=1,padding=1,bias=add_bias),  #(16-3+2)/1+1=16  16*16*128
#         nn.ReLU(inplace=True),

#         nn.Conv2d(in_channels=channels_list[1],out_channels=channels_list[1],kernel_size=3,stride=1,padding=1,bias=add_bias), #(16-3+2)/1+1=16   16*16*128
#         nn.ReLU(inplace=True),
#         #nn.MaxPool2d(2,2)    #(16-2)/2+1=8     8*8*128
#         nn.AvgPool2d(2,2)
#     )

#     layer3=nn.Sequential(
#         nn.Conv2d(in_channels=channels_list[1],out_channels=channels_list[2],kernel_size=3,stride=1,padding=1,bias=add_bias),  #(8-3+2)/1+1=8   8*8*256
#         nn.ReLU(inplace=True),

#         nn.Conv2d(in_channels=channels_list[2],out_channels=channels_list[2],kernel_size=3,stride=1,padding=1,bias=add_bias),  #(8-3+2)/1+1=8   8*8*256
#         nn.ReLU(inplace=True),

#         nn.Conv2d(in_channels=channels_list[2],out_channels=channels_list[2],kernel_size=3,stride=1,padding=1,bias=add_bias),  #(8-3+2)/1+1=8   8*8*256
#         nn.ReLU(inplace=True),

#         #nn.MaxPool2d(2,2)     #(8-2)/2+1=4      4*4*256
#         nn.AvgPool2d(2,2)
#     )

#     layer4=nn.Sequential(
#         nn.Conv2d(in_channels=channels_list[2],out_channels=channels_list[3],kernel_size=3,stride=1,padding=1,bias=add_bias),  #(4-3+2)/1+1=4    4*4*512
#         nn.ReLU(inplace=True),

#         nn.Conv2d(in_channels=channels_list[3],out_channels=channels_list[3],kernel_size=3,stride=1,padding=1,bias=add_bias),   #(4-3+2)/1+1=4    4*4*512
#         nn.ReLU(inplace=True),

#         nn.Conv2d(in_channels=channels_list[3],out_channels=channels_list[3],kernel_size=3,stride=1,padding=1,bias=add_bias),   #(4-3+2)/1+1=4    4*4*512
#         nn.ReLU(inplace=True),

#         #nn.MaxPool2d(2,2)    #(4-2)/2+1=2     2*2*512
#         nn.AvgPool2d(2,2)
#     )

#     layer5=nn.Sequential(
#         nn.Conv2d(in_channels=channels_list[3],out_channels=channels_list[4],kernel_size=3,stride=1,padding=1,bias=add_bias),   #(2-3+2)/1+1=2    2*2*512
#         nn.ReLU(inplace=True),

#         nn.Conv2d(in_channels=channels_list[4],out_channels=channels_list[4],kernel_size=3,stride=1,padding=1,bias=add_bias),  #(2-3+2)/1+1=2     2*2*512
#         nn.ReLU(inplace=True),

#         nn.Conv2d(in_channels=channels_list[4],out_channels=channels_list[4],kernel_size=3,stride=1,padding=1,bias=add_bias),  #(2-3+2)/1+1=2      2*2*512
#         nn.ReLU(inplace=True),

#     )

#     if block_num == 5:
#         conv=nn.Sequential(
#             layer1,
#             layer2,
#             layer3,
#             layer4,
#             layer5
#         )
#     if block_num == 4:
#         conv=nn.Sequential(
#             layer1,
#             layer2,
#             layer3,
#             layer5
#         )
#     if block_num == 3:
#         conv=nn.Sequential(
#             layer1,
#             layer2,
#             layer5
#         )
#     if block_num == 2:
#         conv=nn.Sequential(
#             layer1,
#             layer5
#         )

#     return conv

class CustomPadConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias, pool_set):
        super(CustomPadConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, bias=bias)
        self.pool_set = pool_set
        
    def forward(self, x):
        # 在右侧和下侧进行padding
        if isinstance(x, tuple):
            y = F.pad(x[0], self.pool_set)
            return self.conv(y),x[1]
        else:
            y = F.pad(x, self.pool_set)
            return self.conv(y)

class PaddingLayer(nn.Module):
    def __init__(self, padding):
        super(PaddingLayer, self).__init__()
        self.padding = padding

    def forward(self, x):
        if isinstance(x, tuple):
            return F.pad(x[0], self.padding),x[1]
        else:
            return F.pad(x, self.padding)

class FloorSTE(t.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return t.floor(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()  # 直接传递梯度

class FloorLayer(nn.Module):
    def forward(self, input):
        if isinstance(input, tuple):
            # print('FloorSTE tuple')
            return FloorSTE.apply(input[0]),input[1]
        else:
            return FloorSTE.apply(input)

def Vgg16_net(channels_list=[64, 96, 128, 256, 256], block_num=5):
    add_bias = False
    layer1=nn.Sequential(
        nn.Conv2d(in_channels=3,out_channels=channels_list[0],kernel_size=3,stride=1,padding=1,bias=add_bias), #(32-3+2)/1+1=32   32*32*64
        nn.ReLU(inplace=True),

        nn.Conv2d(in_channels=channels_list[0],out_channels=channels_list[0],kernel_size=3,stride=1,padding=0,bias=add_bias), 
        nn.ReLU(inplace=True),
        #nn.MaxPool2d(kernel_size=2,stride=2)
        nn.AvgPool2d(kernel_size=2,stride=2),
        FloorLayer()
    )

    layer2=nn.Sequential(
        PaddingLayer((0, 1, 1, 1)),
        #CustomPadConv2d(in_channels=channels_list[0], out_channels=channels_list[1], kernel_size=3, stride=1, bias=add_bias, pool_set=(0,1,0,1)),
        nn.Conv2d(in_channels=channels_list[0],out_channels=channels_list[1],kernel_size=3,stride=1,padding=0,bias=add_bias),  #(16-3+2)/1+1=16  16*16*128
        nn.ReLU(inplace=True),

        nn.Conv2d(in_channels=channels_list[1],out_channels=channels_list[1],kernel_size=3,stride=1,padding=1,bias=add_bias), #(16-3+2)/1+1=16   16*16*128
        nn.ReLU(inplace=True),
        #nn.MaxPool2d(kernel_size=2,stride=2)
        nn.AvgPool2d(2,2),
        FloorLayer()
    )

    layer3=nn.Sequential(
        nn.Conv2d(in_channels=channels_list[1],out_channels=channels_list[2],kernel_size=3,stride=1,padding=1,bias=add_bias),  #(8-3+2)/1+1=8   8*8*256
        nn.ReLU(inplace=True),

        nn.Conv2d(in_channels=channels_list[2],out_channels=channels_list[2],kernel_size=3,stride=1,padding=1,bias=add_bias),  #(8-3+2)/1+1=8   8*8*256
        nn.ReLU(inplace=True),

        nn.Conv2d(in_channels=channels_list[2],out_channels=channels_list[2],kernel_size=3,stride=1,padding=1,bias=add_bias),  #(8-3+2)/1+1=8   8*8*256
        nn.ReLU(inplace=True),
        #nn.MaxPool2d(kernel_size=2,stride=2)
        nn.AvgPool2d(2,2),
        FloorLayer()
    )

    layer4=nn.Sequential(
        nn.Conv2d(in_channels=channels_list[2],out_channels=channels_list[3],kernel_size=3,stride=1,padding=1,bias=add_bias),  #(4-3+2)/1+1=4    4*4*512
        nn.ReLU(inplace=True),

        nn.Conv2d(in_channels=channels_list[3],out_channels=channels_list[3],kernel_size=3,stride=1,padding=1,bias=add_bias),   #(4-3+2)/1+1=4    4*4*512
        nn.ReLU(inplace=True),

        nn.Conv2d(in_channels=channels_list[3],out_channels=channels_list[3],kernel_size=3,stride=1,padding=1,bias=add_bias),   #(4-3+2)/1+1=4    4*4*512
        nn.ReLU(inplace=True),
        #nn.MaxPool2d(kernel_size=2,stride=2)
        nn.AvgPool2d(2,2),
        FloorLayer()
    )

    layer5=nn.Sequential(
        nn.Conv2d(in_channels=channels_list[3],out_channels=channels_list[4],kernel_size=3,stride=1,padding=1,bias=add_bias),   #(2-3+2)/1+1=2    2*2*512
        nn.ReLU(inplace=True),

        nn.Conv2d(in_channels=channels_list[4],out_channels=channels_list[4],kernel_size=3,stride=1,padding=1,bias=add_bias),  #(2-3+2)/1+1=2     2*2*512
        nn.ReLU(inplace=True),

        nn.Conv2d(in_channels=channels_list[4],out_channels=channels_list[4],kernel_size=3,stride=1,padding=1,bias=add_bias),  #(2-3+2)/1+1=2      2*2*512
        nn.ReLU(inplace=True),

    )

    if block_num == 5:
        conv=nn.Sequential(
            layer1,
            layer2,
            layer3,
            layer4,
            layer5
        )
    if block_num == 4:
        conv=nn.Sequential(
            layer1,
            layer2,
            layer3,
            layer5
        )
    if block_num == 3:
        conv=nn.Sequential(
            layer1,
            layer2,
            layer5
        )
    if block_num == 2:
        conv=nn.Sequential(
            layer1,
            layer5
        )

    return conv


def vgg16_7():
    fc=nn.Sequential(
        nn.Linear(7*7*256,256,bias=ADDBIAS),
        nn.ReLU(inplace=True),

        nn.Linear(256,256,bias=ADDBIAS),
        nn.ReLU(inplace=True),
    )
    return fc

def vgg16_3():
    fc=nn.Sequential(
        nn.Linear(3*3*256,256,bias=ADDBIAS),
        nn.ReLU(inplace=True),

        nn.Linear(256,256,bias=ADDBIAS),
        nn.ReLU(inplace=True),
    )
    return fc

def vgg16_3_64():
    fc=nn.Sequential(
        nn.Linear(3*3*64,64,bias=ADDBIAS),
        nn.ReLU(inplace=True),

        nn.Linear(64,64,bias=ADDBIAS),
        nn.ReLU(inplace=True),
    )
    return fc

def vgg16_3_32():
    fc=nn.Sequential(
        nn.Linear(3*3*32,32,bias=False),
        nn.ReLU(inplace=True),

        nn.Linear(32,32,bias=False),
        nn.ReLU(inplace=True),
    )
    return fc

def vgg16_4():
    fc=nn.Sequential(
        nn.Linear(4*4*256,256,bias=ADDBIAS),
        nn.ReLU(inplace=True),

        nn.Linear(256,256,bias=ADDBIAS),
        nn.ReLU(inplace=True),
    )
    return fc



def decom_vgg16():
    # the 30th layer of features is relu of conv5_3
    if opt.caffe_pretrain:
        model = vgg16(pretrained=False)
        if not opt.load_path:
            model.load_state_dict(t.load(opt.caffe_pretrain_path))
    else:
        model = vgg16(pretrained=False) # model = vgg16(not opt.load_path)

    features = list(model.features)[:30]
    classifier = model.classifier

    classifier = list(classifier)
    del classifier[6]
    if not opt.use_drop:
        del classifier[5]
        del classifier[2]
    classifier = nn.Sequential(*classifier)

    # freeze top4 conv
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False

    return nn.Sequential(*features), classifier


class FasterRCNNVGG16LIGHT(FasterRCNN):
    """Faster R-CNN based on VGG-16.
    For descriptions on the interface of this model, please refer to
    :class:`model.faster_rcnn.FasterRCNN`.

    Args:
        n_fg_class (int): The number of classes excluding the background.
        ratios (list of floats): This is ratios of width to height of
            the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference
            window.

    """

    feat_stride = 16  # downsample 16x for output of conv5 in vgg16

    def __init__(self,
                 n_fg_class=1,
                 ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32]
                 ):
                 
        extractor = Vgg16_net()
        classifier = vgg16_7()

        rpn = RegionProposalNetwork(
            256, 256,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride,
        )

        head = VGG16RoIHeadlight(
            n_class=n_fg_class + 1,
            roi_size=7,
            spatial_scale=(1. / self.feat_stride),
            classifier=classifier
        )

        super(FasterRCNNVGG16LIGHT, self).__init__(
            extractor,
            rpn,
            head,
        )


class FasterRCNNVGG16LIGHTV2(FasterRCNN):
    """Faster R-CNN based on VGG-16.
    For descriptions on the interface of this model, please refer to
    :class:`model.faster_rcnn.FasterRCNN`.

    Args:
        n_fg_class (int): The number of classes excluding the background.
        ratios (list of floats): This is ratios of width to height of
            the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference
            window.

    """

    feat_stride = 16  # downsample 16x for output of conv5 in vgg16

    def __init__(self,
                use_resnet=False,
                use_maxpool=False,
                use_conv=False,
                use_rois_s=False,
                 n_fg_class=1,
                 ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32]
                 ):
                 
        if use_resnet:
            from .resnet import resnet18
            extractor = resnet18()
        else:
            extractor = Vgg16_net()

        if use_maxpool or use_conv:
            classifier = vgg16_3()
        elif use_rois_s:
            classifier = vgg16_3()
        else:
            classifier = vgg16_7()

        
        rpn = RegionProposalNetwork(
            256, 256,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride,
        )

        if use_maxpool:
            head = VGG16RoIHeadlightPool(
                n_class=n_fg_class + 1,
                roi_size=7,
                spatial_scale=(1. / self.feat_stride),
                classifier=classifier
            )
        elif use_conv:
            head = VGG16RoIHeadlightConv(
                n_class=n_fg_class + 1,
                roi_size=7,
                spatial_scale=(1. / self.feat_stride),
                classifier=classifier
            )
        elif use_rois_s:
            head = VGG16RoIHeadlightRoi(
                n_class=n_fg_class + 1,
                roi_size=3,
                spatial_scale=(1. / self.feat_stride),
                classifier=classifier
            )
        else:
            head = VGG16RoIHeadlight(
                n_class=n_fg_class + 1,
                roi_size=7,
                spatial_scale=(1. / self.feat_stride),
                classifier=classifier
            )

        super(FasterRCNNVGG16LIGHTV2, self).__init__(
            extractor,
            rpn,
            head,
        )

class FasterRCNNVGG16LIGHTV3(FasterRCNN):
    """Faster R-CNN based on VGG-16.
    For descriptions on the interface of this model, please refer to
    :class:`model.faster_rcnn.FasterRCNN`.

    Args:
        n_fg_class (int): The number of classes excluding the background.
        ratios (list of floats): This is ratios of width to height of
            the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference
            window.

    """
    feat_stride = 16
     # downsample 16x for output of conv5 in vgg16

    def __init__(self,
                use_resnet=False,
                use_maxpool=False,
                use_conv=False,
                use_rois_s=False,
                n_fg_class=1,
                ratios=[0.5, 1, 2],
                anchor_scales=[8, 16, 32],
                channel = 64,
                block_num = 5,
                 ):
                 
        self.feat_stride = 16 
        if block_num == 4:
            self.feat_stride = 8
        if block_num == 3:
            self.feat_stride = 4
        if block_num == 2:
            self.feat_stride = 2

        if channel == 64:
            extractor = Vgg16_net(channels_list=[64, 64, 64, 64, 64], block_num=block_num)
            classifier = vgg16_3_64()
            bias = ADDBIAS
            rpn = RegionProposalNetwork(
                64, 32,
                ratios=ratios,
                anchor_scales=anchor_scales,
                feat_stride=self.feat_stride,
            )
        if channel == 32:
            extractor = Vgg16_net(channels_list=[32, 32, 32, 32, 32],block_num=block_num)
            classifier = vgg16_3_32()
            bias = False
            rpn = RegionProposalNetwork(
                32, 32,
                ratios=ratios,
                anchor_scales=anchor_scales,
                feat_stride=self.feat_stride,
                bias = bias
            )
            

        head = VGG16RoIHeadlightRoiV3(
            n_class=n_fg_class + 1,
            roi_size=3,
            spatial_scale=(1. / self.feat_stride),
            classifier=classifier, 
            channel = channel,
            bias = bias
        )
        
        super(FasterRCNNVGG16LIGHTV3, self).__init__(
            extractor,
            rpn,
            head,
        )

class FasterRCNNVGG16(FasterRCNN):
    """Faster R-CNN based on VGG-16.
    For descriptions on the interface of this model, please refer to
    :class:`model.faster_rcnn.FasterRCNN`.

    Args:
        n_fg_class (int): The number of classes excluding the background.
        ratios (list of floats): This is ratios of width to height of
            the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference
            window.

    """

    feat_stride = 16  # downsample 16x for output of conv5 in vgg16

    def __init__(self,
                 n_fg_class=1,
                 ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32]
                 ):
                 
        extractor, classifier = decom_vgg16()

        rpn = RegionProposalNetwork(
            512, 512,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride,
        )

        head = VGG16RoIHead(
            n_class=n_fg_class + 1,
            roi_size=7,
            spatial_scale=(1. / self.feat_stride),
            classifier=classifier
        )

        super(FasterRCNNVGG16, self).__init__(
            extractor,
            rpn,
            head,
        )


class VGG16RoIHeadlight(nn.Module):
    """Faster R-CNN Head for VGG-16 based implementation.
    This class is used as a head for Faster R-CNN.
    This outputs class-wise localizations and classification based on feature
    maps in the given RoIs.
    
    Args:
        n_class (int): The number of classes possibly including the background.
        roi_size (int): Height and width of the feature maps after RoI-pooling.
        spatial_scale (float): Scale of the roi is resized.
        classifier (nn.Module): Two layer Linear ported from vgg16

    """

    def __init__(self, n_class, roi_size, spatial_scale,
                 classifier):
        # n_class includes the background
        super(VGG16RoIHeadlight, self).__init__()

        self.classifier = classifier
        self.cls_loc = nn.Linear(256, n_class * 4)
        self.score = nn.Linear(256, n_class)

        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.roi = RoIPool( (self.roi_size, self.roi_size),self.spatial_scale)
    
    def forward(self, x, rois, roi_indices):
        """Forward the chain.

        We assume that there are :math:`N` batches.

        Args:
            x (Variable): 4D image variable.
            rois (Tensor): A bounding box array containing coordinates of
                proposal boxes.  This is a concatenation of bounding box
                arrays from multiple images in the batch.
                Its shape is :math:`(R', 4)`. Given :math:`R_i` proposed
                RoIs from the :math:`i` th image,
                :math:`R' = \\sum _{i=1} ^ N R_i`.
            roi_indices (Tensor): An array containing indices of images to
                which bounding boxes correspond to. Its shape is :math:`(R',)`.

        """
        # in case roi_indices is  ndarray
        
        roi_indices = at.totensor(roi_indices).float()
        rois = at.totensor(rois).float()
        indices_and_rois = t.cat([roi_indices[:, None], rois], dim=1)
        # NOTE: important: yx->xy
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois =  xy_indices_and_rois.contiguous()

        pool = self.roi(x, indices_and_rois)
        
        pool = pool.view(pool.size(0), -1)

        fc7 = self.classifier(pool)
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)
        return roi_cls_locs, roi_scores


class VGG16RoIHeadlightPool(nn.Module):
    """Faster R-CNN Head for VGG-16 based implementation.
    This class is used as a head for Faster R-CNN.
    This outputs class-wise localizations and classification based on feature
    maps in the given RoIs.
    
    Args:
        n_class (int): The number of classes possibly including the background.
        roi_size (int): Height and width of the feature maps after RoI-pooling.
        spatial_scale (float): Scale of the roi is resized.
        classifier (nn.Module): Two layer Linear ported from vgg16

    """

    def __init__(self, n_class, roi_size, spatial_scale,
                 classifier):
        # n_class includes the background
        super(VGG16RoIHeadlightPool, self).__init__()

        self.classifier = classifier
        self.cls_loc = nn.Linear(256, n_class * 4)
        self.score = nn.Linear(256, n_class)

        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.roi = RoIPool( (self.roi_size, self.roi_size),self.spatial_scale)

        self.maxpool = nn.MaxPool2d(2,2)

    def forward(self, x, rois, roi_indices):
        """Forward the chain.

        We assume that there are :math:`N` batches.

        Args:
            x (Variable): 4D image variable.
            rois (Tensor): A bounding box array containing coordinates of
                proposal boxes.  This is a concatenation of bounding box
                arrays from multiple images in the batch.
                Its shape is :math:`(R', 4)`. Given :math:`R_i` proposed
                RoIs from the :math:`i` th image,
                :math:`R' = \\sum _{i=1} ^ N R_i`.
            roi_indices (Tensor): An array containing indices of images to
                which bounding boxes correspond to. Its shape is :math:`(R',)`.

        """
        # in case roi_indices is  ndarray
        
        roi_indices = at.totensor(roi_indices).float()
        rois = at.totensor(rois).float()
        indices_and_rois = t.cat([roi_indices[:, None], rois], dim=1)
        # NOTE: important: yx->xy
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois =  xy_indices_and_rois.contiguous()

      
        pool = self.roi(x, indices_and_rois)
        pool = self.maxpool(pool)

        pool = pool.view(pool.size(0), -1)

        fc7 = self.classifier(pool)
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)
        return roi_cls_locs, roi_scores

class VGG16RoIHeadlightRoi(nn.Module):
    """Faster R-CNN Head for VGG-16 based implementation.
    This class is used as a head for Faster R-CNN.
    This outputs class-wise localizations and classification based on feature
    maps in the given RoIs.
    
    Args:
        n_class (int): The number of classes possibly including the background.
        roi_size (int): Height and width of the feature maps after RoI-pooling.
        spatial_scale (float): Scale of the roi is resized.
        classifier (nn.Module): Two layer Linear ported from vgg16

    """

    def __init__(self, n_class, roi_size, spatial_scale,
                 classifier):
        # n_class includes the background
        super(VGG16RoIHeadlightRoi, self).__init__()

        self.classifier = classifier
        self.cls_loc = nn.Linear(256, n_class * 4)
        self.score = nn.Linear(256, n_class)

        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.roi = RoIPool( (self.roi_size, self.roi_size),self.spatial_scale)
        
    def forward(self, x, rois, roi_indices):
        """Forward the chain.

        We assume that there are :math:`N` batches.

        Args:
            x (Variable): 4D image variable.
            rois (Tensor): A bounding box array containing coordinates of
                proposal boxes.  This is a concatenation of bounding box
                arrays from multiple images in the batch.
                Its shape is :math:`(R', 4)`. Given :math:`R_i` proposed
                RoIs from the :math:`i` th image,
                :math:`R' = \\sum _{i=1} ^ N R_i`.
            roi_indices (Tensor): An array containing indices of images to
                which bounding boxes correspond to. Its shape is :math:`(R',)`.

        """
        # in case roi_indices is  ndarray
        
        roi_indices = at.totensor(roi_indices).float()
        rois = at.totensor(rois).float()
        indices_and_rois = t.cat([roi_indices[:, None], rois], dim=1)
        # NOTE: important: yx->xy
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois =  xy_indices_and_rois.contiguous()

      
        pool = self.roi(x, indices_and_rois)

        pool = pool.view(pool.size(0), -1)

        fc7 = self.classifier(pool)
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)
        return roi_cls_locs, roi_scores

class VGG16RoIHeadlightRoiV3(nn.Module):
    """Faster R-CNN Head for VGG-16 based implementation.
    This class is used as a head for Faster R-CNN.
    This outputs class-wise localizations and classification based on feature
    maps in the given RoIs.
    
    Args:
        n_class (int): The number of classes possibly including the background.
        roi_size (int): Height and width of the feature maps after RoI-pooling.
        spatial_scale (float): Scale of the roi is resized.
        classifier (nn.Module): Two layer Linear ported from vgg16

    """

    def __init__(self, n_class, roi_size, spatial_scale,
                 classifier, channel=64, bias=True):
        # n_class includes the background
        super(VGG16RoIHeadlightRoiV3, self).__init__()

        self.classifier = classifier
        self.cls_loc = nn.Linear(channel, n_class * 4, bias=bias)
        self.score = nn.Linear(channel, n_class,bias=bias)

        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.roi = RoIPool( (self.roi_size, self.roi_size),self.spatial_scale)

        

    def forward(self, x, rois, roi_indices):
        """Forward the chain.

        We assume that there are :math:`N` batches.

        Args:
            x (Variable): 4D image variable.
            rois (Tensor): A bounding box array containing coordinates of
                proposal boxes.  This is a concatenation of bounding box
                arrays from multiple images in the batch.
                Its shape is :math:`(R', 4)`. Given :math:`R_i` proposed
                RoIs from the :math:`i` th image,
                :math:`R' = \\sum _{i=1} ^ N R_i`.
            roi_indices (Tensor): An array containing indices of images to
                which bounding boxes correspond to. Its shape is :math:`(R',)`.

        """
        # in case roi_indices is  ndarray
        
        roi_indices = at.totensor(roi_indices).float()
        rois = at.totensor(rois).float()
        indices_and_rois = t.cat([roi_indices[:, None], rois], dim=1)
        # NOTE: important: yx->xy
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois =  xy_indices_and_rois.contiguous()
        if len(x)==2:
            # print('ROI scale x[1]:',x[1])
            x = x[0]*x[1]
        pool = self.roi(x, indices_and_rois)

        pool = pool.view(pool.size(0), -1)

        fc7 = self.classifier(pool)
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)
        return roi_cls_locs, roi_scores

class VGG16RoIHeadlightConv(nn.Module):
    """Faster R-CNN Head for VGG-16 based implementation.
    This class is used as a head for Faster R-CNN.
    This outputs class-wise localizations and classification based on feature
    maps in the given RoIs.
    
    Args:
        n_class (int): The number of classes possibly including the background.
        roi_size (int): Height and width of the feature maps after RoI-pooling.
        spatial_scale (float): Scale of the roi is resized.
        classifier (nn.Module): Two layer Linear ported from vgg16

    """

    def __init__(self, n_class, roi_size, spatial_scale,
                 classifier):
        # n_class includes the background
        super(VGG16RoIHeadlightConv, self).__init__()

        self.classifier = classifier
        self.cls_loc = nn.Linear(256, n_class * 4,bias=ADDBIAS)
        self.score = nn.Linear(256, n_class,bias=ADDBIAS)

        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.roi = RoIPool( (self.roi_size, self.roi_size),self.spatial_scale)

        self.conv = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=2,padding=0)


    def forward(self, x, rois, roi_indices):
        """Forward the chain.

        We assume that there are :math:`N` batches.

        Args:
            x (Variable): 4D image variable.
            rois (Tensor): A bounding box array containing coordinates of
                proposal boxes.  This is a concatenation of bounding box
                arrays from multiple images in the batch.
                Its shape is :math:`(R', 4)`. Given :math:`R_i` proposed
                RoIs from the :math:`i` th image,
                :math:`R' = \\sum _{i=1} ^ N R_i`.
            roi_indices (Tensor): An array containing indices of images to
                which bounding boxes correspond to. Its shape is :math:`(R',)`.

        """
        # in case roi_indices is  ndarray
        
        roi_indices = at.totensor(roi_indices).float()
        rois = at.totensor(rois).float()
        indices_and_rois = t.cat([roi_indices[:, None], rois], dim=1)
        # NOTE: important: yx->xy
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois =  xy_indices_and_rois.contiguous()
       
        pool = self.roi(x, indices_and_rois)
        pool = self.conv(pool)
        
        pool = pool.view(pool.size(0), -1)

        fc7 = self.classifier(pool)
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)
        return roi_cls_locs, roi_scores

class VGG16RoIHead(nn.Module):
    """Faster R-CNN Head for VGG-16 based implementation.
    This class is used as a head for Faster R-CNN.
    This outputs class-wise localizations and classification based on feature
    maps in the given RoIs.
    
    Args:
        n_class (int): The number of classes possibly including the background.
        roi_size (int): Height and width of the feature maps after RoI-pooling.
        spatial_scale (float): Scale of the roi is resized.
        classifier (nn.Module): Two layer Linear ported from vgg16

    """

    def __init__(self, n_class, roi_size, spatial_scale,
                 classifier):
        # n_class includes the background
        super(VGG16RoIHead, self).__init__()

        self.classifier = classifier
        self.cls_loc = nn.Linear(4096, n_class * 4,bias=ADDBIAS)
        self.score = nn.Linear(4096, n_class,bias=ADDBIAS)

        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.roi = RoIPool( (self.roi_size, self.roi_size),self.spatial_scale)

    def forward(self, x, rois, roi_indices):
        """Forward the chain.

        We assume that there are :math:`N` batches.

        Args:
            x (Variable): 4D image variable.
            rois (Tensor): A bounding box array containing coordinates of
                proposal boxes.  This is a concatenation of bounding box
                arrays from multiple images in the batch.
                Its shape is :math:`(R', 4)`. Given :math:`R_i` proposed
                RoIs from the :math:`i` th image,
                :math:`R' = \\sum _{i=1} ^ N R_i`.
            roi_indices (Tensor): An array containing indices of images to
                which bounding boxes correspond to. Its shape is :math:`(R',)`.

        """
        # in case roi_indices is  ndarray
        
        roi_indices = at.totensor(roi_indices).float()
        rois = at.totensor(rois).float()
        indices_and_rois = t.cat([roi_indices[:, None], rois], dim=1)
        # NOTE: important: yx->xy
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois =  xy_indices_and_rois.contiguous()

        pool = self.roi(x, indices_and_rois)
        pool = pool.view(pool.size(0), -1)
        fc7 = self.classifier(pool)
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)
        return roi_cls_locs, roi_scores


def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        # if ADDBIAS:
        #     m.bias.data.zero_()
