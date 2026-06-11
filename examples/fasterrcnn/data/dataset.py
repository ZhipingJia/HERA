from __future__ import  absolute_import
from __future__ import  division
import torch as t
from data.voc_dataset import VOCBboxDataset
from skimage import transform as sktsf
from torchvision import transforms as tvtsf
from data import util
import numpy as np
from utils.config import opt
import random
import json
import time

def inverse_normalize(img):
    if opt.caffe_pretrain:
        img = img + (np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1))
        return img[::-1, :, :]
    # approximate un-normalize for visualize
    return (img * 0.225 + 0.45).clip(min=0, max=1) * 255


def pytorch_normalze(img):
    """
    https://github.com/pytorch/vision/issues/223
    return appr -1~1 RGB
    """
    normalize = tvtsf.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    img = normalize(t.from_numpy(img))
    return img.numpy()


def caffe_normalize(img):
    """
    return appr -125-125 BGR
    """
    img = img[[2, 1, 0], :, :]  # RGB-BGR
    img = img * 255
    mean = np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1)
    img = (img - mean).astype(np.float32, copy=True)
    return img


def preprocess(img, min_size=600, max_size=1000):
    """Preprocess an image for feature extraction.

    The length of the shorter edge is scaled to :obj:`self.min_size`.
    After the scaling, if the length of the longer edge is longer than
    :param min_size:
    :obj:`self.max_size`, the image is scaled to fit the longer edge
    to :obj:`self.max_size`.

    After resizing the image, the image is subtracted by a mean image value
    :obj:`self.mean`.

    Args:
        img (~numpy.ndarray): An image. This is in CHW and RGB format.
            The range of its value is :math:`[0, 255]`.

    Returns:
        ~numpy.ndarray: A preprocessed image.

    """
    C, H, W = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    #print('scale:',scale)
    img = img / 255.
    img = sktsf.resize(img, (C, H * scale, W * scale), mode='reflect',anti_aliasing=False)
    # both the longer and shorter should be less than
    # max_size and min_size
    if opt.caffe_pretrain:
        normalize = caffe_normalize
    else:
        normalize = pytorch_normalze
    return normalize(img)


class Transform(object):

    def __init__(self, min_size=600, max_size=1000):
    
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, in_data):
        img, bbox, label = in_data
        _, H, W = img.shape
        #print(H,W)
        img = preprocess(img, self.min_size, self.max_size)
        #print('tsfing:',img.shape)
        _, o_H, o_W = img.shape
        scale = o_H / H
        bbox = util.resize_bbox(bbox, (H, W), (o_H, o_W))

        # horizontally flip
        img, params = util.random_flip(
            img, x_random=True, return_param=True)
        bbox = util.flip_bbox(
            bbox, (o_H, o_W), x_flip=params['x_flip'])

        return img, bbox, label, scale

class Slice(object):
    def __init__(self, max_info):
        self.max_info = max_info
        self.start = 0
        self.stop = 0

    def valid_choice(self):
        if self.max_info == 960:
            self.start = 190
            self.stop = 960
        if self.max_info == 1280:
            self.start = 120
            self.stop = 1080

def valid_crop_fun(ori_copy, bbox_copy, y_slice, x_slice):
    ori_img, bbox = ori_copy.copy(),bbox_copy.copy()
    ori_img = ori_img[:,y_slice.start:y_slice.stop, x_slice.start:x_slice.stop]
    #print(ori_img.shape)
    bbox = util.crop_bbox(bbox, y_slice, x_slice)
    return ori_img, bbox


def fixed_crop_fun(ori_copy, bbox_copy, y_slice, x_slice, slice_info):
    ori_img, bbox = ori_copy.copy(),bbox_copy.copy()
    x_slice = Slice(1280)
    y_slice = Slice(960)
    x_slice.start, x_slice.stop, y_slice.start, y_slice.stop = slice_info
    ori_img = ori_img[:,y_slice.start:y_slice.stop, x_slice.start:x_slice.stop]
    bbox = util.crop_bbox(bbox, y_slice, x_slice)
    return ori_img, bbox


class Dataset:
    def __init__(self, opt):
        self.opt = opt
        self.db = VOCBboxDataset(opt.voc_data_dir)
        self.tsf = Transform(opt.min_size, opt.max_size)
        self.crop = opt.crop

        self.x_slice = Slice(1280)
        self.y_slice = Slice(960)
        self.remove_list = []

        
        if self.crop == 'valid_crop':
            self.x_slice.valid_choice()
            self.y_slice.valid_choice()
            
            for idx,id in enumerate(self.db.ids):
                ori_img, bbox, label, difficult, img_id = self.db.get_example(idx)
                ori_img, bbox = valid_crop_fun(ori_img, bbox, self.y_slice, self.x_slice)
                if len(bbox)==0:
                    self.remove_list.append(id)
    
        if self.crop == 'fixed_crop':
            self.crop_h = opt.crop_h
            self.crop_w = opt.crop_w
            f = open(f"{opt.voc_data_dir}/slice_info_w{self.crop_w}_h{self.crop_h}.json","r")
            self.slice_dict = json.load(f)
            for idx,id in enumerate(self.db.ids):
                ori_img, bbox, label, difficult, img_id = self.db.get_example(idx)
                if (bbox[:,2]-bbox[:,0]).max() >= self.crop_h or (bbox[:,3]-bbox[:,1]).max() >= self.crop_w or img_id not in self.slice_dict:
                    self.remove_list.append(id)

        _ = [self.db.ids.remove(id) for id in self.remove_list]
        print('train', len(self.db.ids))

    def __getitem__(self, idx):
        ori_img, bbox, label, difficult, img_id = self.db.get_example(idx)
        
        
        if self.crop == 'valid_crop':
            ori_img, bbox = valid_crop_fun(ori_img, bbox, self.y_slice, self.x_slice)
            label = label[:len(bbox)]
            difficult = difficult[:len(bbox)]
    
        if self.crop == 'fixed_crop':
            ori_img, bbox = fixed_crop_fun(ori_img, bbox, self.y_slice, self.x_slice, self.slice_dict[img_id])
            label = label[:len(bbox)]
            difficult = difficult[:len(bbox)]
        #print(ori_img.shape)
        img, bbox, label, scale = self.tsf((ori_img, bbox, label))
        #print('train shape:',img.shape)
        # TODO: check whose stride is negative to fix this instead copy all
        # some of the strides of a given numpy array are negative.
        return img.copy(), ori_img.shape[1:], bbox.copy(), label.copy(), scale

    def __len__(self):
        return len(self.db)


class TestDataset:
    def __init__(self, opt, split='test', use_difficult=True):
        self.opt = opt
        self.min_size = opt.min_size
        self.max_size = opt.max_size
        self.db = VOCBboxDataset(opt.voc_data_dir, split=split, use_difficult=use_difficult)
        self.crop = opt.crop
        
        self.x_slice = Slice(1280)
        self.y_slice = Slice(960)
        self.remove_list = []

        if self.crop == 'valid_crop':
            self.x_slice.valid_choice()
            self.y_slice.valid_choice()
            
            for idx,id in enumerate(self.db.ids):
                ori_img, bbox, label, difficult, img_id = self.db.get_example(idx)
                ori_img, bbox = valid_crop_fun(ori_img, bbox, self.y_slice, self.x_slice)
                if len(bbox)==0:
                    self.remove_list.append(id)
    
        if self.crop == 'fixed_crop':
            self.crop_h = opt.crop_h
            self.crop_w = opt.crop_w
            f = open(f"{opt.voc_data_dir}/slice_info_w{self.crop_w}_h{self.crop_h}.json","r")
            self.slice_dict = json.load(f)
            for idx,id in enumerate(self.db.ids):
                ori_img, bbox, label, difficult, img_id = self.db.get_example(idx)
                if (bbox[:,2]-bbox[:,0]).max() >= self.crop_h or (bbox[:,3]-bbox[:,1]).max() >= self.crop_w or img_id not in self.slice_dict:
                    self.remove_list.append(id)
    
        _ = [self.db.ids.remove(id) for id in self.remove_list]
        
        print('test',len(self.db.ids))
        
    def __getitem__(self, idx):
        ori_img, bbox, label, difficult, img_id = self.db.get_example(idx)
        # label = label[:len(bbox)]
        # difficult = difficult[:len(bbox)]
        # print(ori_img.shape)
        # time0 = time.time()
        if self.crop == 'valid_crop':
            ori_img, bbox = valid_crop_fun(ori_img, bbox, self.y_slice, self.x_slice)
            label = label[:len(bbox)]
            difficult = difficult[:len(bbox)]
        #time1 = time.time()
        if self.crop == 'fixed_crop':
            ori_img, bbox = fixed_crop_fun(ori_img, bbox, self.y_slice, self.x_slice, self.slice_dict[img_id])
            label = label[:len(bbox)]
            difficult = difficult[:len(bbox)]
        # print('croped shape:',ori_img.shape)
        # print('bbox:',bbox)
        img = preprocess(ori_img,self.min_size, self.max_size)
        _, H, W = ori_img.shape
        _, o_H, o_W = img.shape
        scale = o_H / H
        #print('preprocessed shape:',img.shape)
        #time2 = time.time()
        # print('crop_time:',time1-time0)
        # print('preprocess_time:',time2-time1)
        #print('test shape:',img.shape)
        return img, ori_img.shape[1:], bbox, label, difficult, scale

    def __len__(self):
        return len(self.db)
