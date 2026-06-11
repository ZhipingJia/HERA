"""Runtime configuration for the HERA Faster R-CNN workload.

This is a sanitized version of the original experiment config.  All dataset and
checkpoint locations must be supplied explicitly (CLI flags of the entry scripts
call ``opt._parse``); no absolute server paths are shipped.  Field names are kept
identical to the original codebase so the migrated model/data code runs unchanged.
"""

from pprint import pprint

import numpy as np


def get_count_loss(module):
    """Count-loss regularizer over INT weight column occupancy (training only)."""
    weight_q_mapping = module.get_int_weight()[0]
    count_loss = ((weight_q_mapping.abs().reshape(32, 32, 9).permute(2, 1, 0).reshape(-1, 32).sum(axis=0) - 160) ** 2).mean()
    return count_loss


def get_count(name, weight_module, logger):
    """Log INT weight column occupancy statistics (training only)."""
    weight_q, weight_scale = weight_module.get_int_weight()
    weight_q_mapping = weight_q.reshape(32, 32, 9).permute(2, 1, 0).reshape(-1, 32)
    nonzero_indices = weight_q_mapping.nonzero()[:, 1]
    count_vector = np.bincount(nonzero_indices.cpu().numpy(), minlength=32)
    logger.info(f"{name} count num {round(np.mean((count_vector - 160) ** 2))} {count_vector}")


class Config:
    # --- data (must be provided by the user; VOC-style layout, see README) ---
    voc_data_dir = None          # root containing JPEGImages/, Annotations/, list_files/
    min_size = 154               # image resize lower bound (paper setting)
    max_size = 1000              # image resize upper bound
    crop = "valid_crop"          # crop mode used by the infrared detection task
    crop_h = 240
    crop_w = 320

    num_workers = 4
    test_num_workers = 4
    test_num = 10000

    # --- model (paper's lightweight channel-32 variant) ---
    light_version = 3
    channel = 32
    block_num = 3
    anchor_scales = [2, 4, 8]
    use_resnet = False
    use_maxpool = False
    use_conv = False
    use_rois_s = True
    use_drop = False
    caffe_pretrain = False
    caffe_pretrain_path = None
    load_path = None             # checkpoint path; set via CLI

    # --- device ---
    device = 0

    # --- optimizer / training fields (referenced by FasterRCNN.get_optimizer and trainer) ---
    lr = 1e-4
    weight_decay = 0.0005
    use_adam = False
    lr_decay = 0.1
    epoch = 0
    decay_epoch = 30
    rpn_sigma = 3.0
    roi_sigma = 1.0
    weight_count_loss = {}

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items()
                if not k.startswith('_')}


opt = Config()
