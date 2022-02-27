import torch
import argparse
import os
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.utils.data as data

import extorch.vision.dataset as dataset
import extorch.utils as utils

from model.resnet import *


def main():
    parser = argparse.ArgumentParser()

    # Basic settings
    parser.add_argument("--load-net1", default='/home/utianjin/BAR/res18_0.2/prune/final.ckpt', type=str,
                        help="load the pretrained network from disk")
    parser.add_argument("--load-net2", default='/home/utianjin/bardiversity/divres18_0.2/prune/final.ckpt', type=str,
                        help="load the pretrained network from disk")
    #parser.add_argument("--load-net3", default='/home/gzsys/桌面/BAR-TEST/net3_1/finetune/final.ckpt', type=str,
    #                    help="load the pretrained network from disk")
    parser.add_argument("--num-workers", default=2, type=int)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--report-every", default=100, type=int)

    # hyper-parameter settings
    parser.add_argument("--epochs", default=80, type=int)
    args = parser.parse_args()

    LOGGER = utils.getLogger("Compare")
    if args.train_dir:
        utils.makedir(args.train_dir, remove = True)
        LOGGER.addFile(os.path.join(args.train_dir, "train.log"))

    for name in vars(args):
        LOGGER.info("{}: {}".format(name, getattr(args, name)))

    DEVICE = torch.device("cuda:{}".format(args.gpu)) \
            if torch.cuda.is_available() else torch.device("cpu")

    if args.seed:
        utils.set_seed(args.seed)
        LOGGER.info("Set seed: {}".format(args.seed))

    # Construct the network
    LOGGER.info("Load checkpoint from {}".format(args.load_net1))
    net1 = torch.load(args.load_net1)
    net1 = net1.to(DEVICE)
    num_params = utils.get_params(net1)
    LOGGER.info("Parameter size: {:.5f}M".format(num_params / 1.e6))

    LOGGER.info("Load checkpoint from {}".format(args.load_net1))
    net2 = torch.load(args.load_net2)
    net2 = net2.to(DEVICE)
    num_params = utils.get_params(net2)
    LOGGER.info("Parameter size: {:.5f}M".format(num_params / 1.e6))
    mask1 = net1.hard_prune()
    mask2 = net2.hard_prune()

    num_one_1 = torch.sum(mask1[0:64 - 1])
    num_one_2 = torch.sum(mask2[0:64 - 1])
    LOGGER.info("numbers of one in net1 con1 or bn1 : num_one_1 {:.0f}".format(num_one_1))
    LOGGER.info("numbers of one in net2 con1 or bn1 : num_one_2 {:.0f}".format(num_one_2))
    count_con1bn1 = 0
    for i in range(64):
        if mask1[i] != mask2[i]:
            count_con1bn1 = count_con1bn1 + 1
    LOGGER.info("numbers of difference in  con1 or bn1 : count_con1bn1 {:.0f}".format(count_con1bn1))

    num_one_1 = torch.sum(mask1[64:64 * 3 - 1])
    num_one_2 = torch.sum(mask2[64:64 * 3 - 1])
    LOGGER.info("numbers of one in net1 block1 of layer1 : num_one_1 {:.0f}".format(num_one_1))
    LOGGER.info("numbers of one in net2 block1 of layer1 : num_one_2 {:.0f}".format(num_one_2))
    count= 0
    for i in range(64, 64 * 3):
        if mask1[i] != mask2[i]:
            count = count + 1
    LOGGER.info("numbers of difference in  block1 of layer1 : count {:.0f}".format(count))

    num_one_1 = torch.sum(mask1[64*3:64 * 5 - 1])
    num_one_2 = torch.sum(mask2[64*3:64 * 5 - 1])
    LOGGER.info("numbers of one in net1 block2 of layer1 : num_one_1 {:.0f}".format(num_one_1))
    LOGGER.info("numbers of one in net2 block2 of layer1 : num_one_2 {:.0f}".format(num_one_2))
    count = 0
    for i in range(64*3, 64 * 5):
        if mask1[i] != mask2[i]:
            count = count + 1
    LOGGER.info("numbers of difference in  block2 of layer1 : count {:.0f}".format(count))

    num_one_1 = torch.sum(mask1[320:783])
    num_one_2 = torch.sum(mask2[320:783])
    LOGGER.info("numbers of one in net1 block1 of layer2 : num_one_1 {:.0f}".format(num_one_1))
    LOGGER.info("numbers of one in net2 block1 of layer2 : num_one_2 {:.0f}".format(num_one_2))
    count = 0
    for i in range(320, 783):
        if mask1[i] != mask2[i]:
            count = count + 1
    LOGGER.info("numbers of difference in  block1 of layer2 : count {:.0f}".format(count))

    num_one_1 = torch.sum(mask1[784:959])
    num_one_2 = torch.sum(mask2[784:959])
    LOGGER.info("numbers of one in net1 block2 of layer2 : num_one_1 {:.0f}".format(num_one_1))
    LOGGER.info("numbers of one in net2 block2 of layer2 : num_one_2 {:.0f}".format(num_one_2))
    count = 0
    for i in range(784, 959):
        if mask1[i] != mask2[i]:
            count = count + 1
    LOGGER.info("numbers of difference in  block2 of layer2 : count {:.0f}".format(count))

    num_one_1 = torch.sum(mask1[960:1727])
    num_one_2 = torch.sum(mask2[960:1727])
    LOGGER.info("numbers of one in net1 block1 of layer3 : num_one_1 {:.0f}".format(num_one_1))
    LOGGER.info("numbers of one in net2 block1 of layer3 : num_one_2 {:.0f}".format(num_one_2))
    count = 0
    for i in range(960, 1727):
        if mask1[i] != mask2[i]:
            count = count + 1
    LOGGER.info("numbers of difference in  block1 of layer3 : count {:.0f}".format(count))

    num_one_1 = torch.sum(mask1[1728:2239])
    num_one_2 = torch.sum(mask2[1728:2239])
    LOGGER.info("numbers of one in net1 block2 of layer3 : num_one_1 {:.0f}".format(num_one_1))
    LOGGER.info("numbers of one in net2 block2 of layer3 : num_one_2 {:.0f}".format(num_one_2))
    count = 0
    for i in range(1728,2239):
        if mask1[i] != mask2[i]:
            count = count + 1
    LOGGER.info("numbers of difference in  block2 of layer3 : count {:.0f}".format(count))

    num_one_1 = torch.sum(mask1[2240:3775])
    num_one_2 = torch.sum(mask2[2240:3775])
    LOGGER.info("numbers of one in net1 block1 of layer4 : num_one_1 {:.0f}".format(num_one_1))
    LOGGER.info("numbers of one in net2 block1 of layer4 : num_one_2 {:.0f}".format(num_one_2))
    count = 0
    for i in range(2240, 3776):
        if mask1[i] != mask2[i]:
            count = count + 1
    LOGGER.info("numbers of difference in  block1 of layer4 : count {:.0f}".format(count))

    num_one_1 = torch.sum(mask1[3776:4799])
    num_one_2 = torch.sum(mask2[3776:4799])
    LOGGER.info("numbers of one in net1 block2 of layer4 : num_one_1 {:.0f}".format(num_one_1))
    LOGGER.info("numbers of one in net2 block2 of layer4 : num_one_2 {:.0f}".format(num_one_2))
    count = 0
    for i in range(3776, 4799):
        if mask1[i] != mask2[i]:
            count = count + 1
    LOGGER.info("numbers of difference in  block2 of layer4 : count {:.0f}".format(count))


if __name__ == "__main__":
    main()










