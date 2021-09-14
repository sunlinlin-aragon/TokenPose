# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
os.environ["CUDA_VISIBLE_DEVICES"] = '4'
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import validate
from utils.utils import create_logger
from tifffile import tifffile
import random
from PIL import Image
import copy
import cv2
import numpy as np
from core.inference import get_final_preds
from utils.utils import show_point
import models
from utils.transforms import flip_back
from utils.transforms import get_affine_transform
from utils.transforms import affine_transform


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=False,
                        type=str,
                        default='/home/workspace/TokenPose/experiments/coco/tokenpose/tokenpose_L_D24_384_288_patch64_dim192_depth24_heads12.yaml'
                        )

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='./ckpts',
                        required=False
                        )
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        required=False,
                        default='./logs')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        required=False,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        required=False,
                        default='')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    model_state_file = os.path.join(
        final_output_dir, 'checkpoint.pth'
    )
    logger.info('=> loading model from {}'.format(model_state_file))
    model.load_state_dict(torch.load(model_state_file)['state_dict'])
    model.cuda()


    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    file_path = '/home/workspace/data/CIHP_body_front_point/val_img/7a7a06fbb9e44a62b3dad4d397349ec7.tif'
    data_numpy = tifffile.imread(file_path)
    data_numpy = np.expand_dims(data_numpy, axis=2)
    data_numpy = np.concatenate((data_numpy, data_numpy, data_numpy), axis=-1)
    data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)
    c = np.array([139.5, 194.5], dtype=np.float32)
    s = np.array([1.546875, 2.0625], dtype=np.float32)
    r = 0
    image_size = np.array([288, 384])
    print(c, s, r, image_size)
    trans = get_affine_transform(c, s, r, image_size)
    input = cv2.warpAffine(data_numpy, trans, (int(image_size[0]), int(image_size[1])), flags=cv2.INTER_LINEAR)
    input = transform(input).unsqueeze(0)
    print(input.shape)
    with torch.no_grad():
        outputs = model(input.cuda())
        if isinstance(outputs, list):
            output = outputs[-1]
        else:
            output = outputs
        # this part is ugly, because pytorch has not supported negative index
        # input_flipped = model(input[:, :, :, ::-1])
        # input_flipped = np.flip(input.cpu().numpy(), 3).copy()
        # input_flipped = torch.from_numpy(input_flipped).cuda()
        # outputs_flipped = model(input_flipped)

        # if isinstance(outputs_flipped, list):
        #     output_flipped = outputs_flipped[-1]
        # else:
        #     output_flipped = outputs_flipped

        # flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8],
        #             [11, 12], [13, 14], [18, 19],
        #             [20, 21], [22, 23], [24, 25],
        #             [26, 27], [28, 29], [30, 31],
        #             [32, 33]
        #         ]
        # output_flipped = flip_back(output_flipped.cpu().numpy(), flip_pairs)
        # output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

        # output = (output + output_flipped) * 0.5

        c = np.array([[139.5, 194.5]], dtype=np.float32)
        s = np.array([[1.546875, 2.0625]], dtype=np.float32)

        preds, maxvals = get_final_preds(cfg, output.clone().cpu().numpy(), c, s)

        image = tifffile.imread(file_path)
        img = copy.copy(image) * 5
        img = show_point(img, preds[0], (0, 0, 255))   # 识别
        Image.fromarray(np.uint8(img)).save('test.png')


if __name__ == '__main__':
    main()
