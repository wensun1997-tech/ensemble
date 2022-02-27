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


def valid(net, testloader, device, report_every, logger):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    net.eval()
    logits_all = torch.tensor([])
    logits_all = logits_all.to(device)
    with torch.no_grad():
        for step, (inputs, labels) in enumerate(testloader):

            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = net(inputs)
            logits_all = torch.cat((logits_all, logits), 0)
            prec1, prec5 = utils.accuracy(logits, labels, topk = (1, 5))
            n = inputs.size(0)

            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if (step + 1) % report_every == 0:
                logger.info("valid {} / {} {:.3f}%; {:.3f}%".format(step + 1, len(testloader), top1.avg, top5.avg))

    return logits_all, top1.avg, top5.avg


def main():
    parser = argparse.ArgumentParser()

    # Basic settings
    parser.add_argument("--data-dir", type=str, default='/home/gzsys/桌面/BAR-TEST/data')
    parser.add_argument("--seed", default=2, type=int)
    parser.add_argument("--gpu", type=int, default=0, help="gpu device id")
    parser.add_argument("--train-dir", type=str, default='/home/gzsys/桌面/BAR-TEST/ensemblediverse_n3',
                        help="path to save the checkpoints")
    parser.add_argument("--load-net1", default='/home/gzsys/桌面/BAR-TEST/prune-models/pretrain/final.ckpt', type=str,
                        help="load the pretrained network from disk")
    parser.add_argument("--load-net2", default='/home/gzsys/桌面/BAR-TEST/t2_6.7/finetune/final.ckpt', type=str,
                        help="load the pretrained network from disk")
    #parser.add_argument("--load-net3", default='/home/gzsys/桌面/BAR-TEST/net3_1/finetune/final.ckpt', type=str,
    #                    help="load the pretrained network from disk")
    parser.add_argument("--num-workers", default=2, type=int)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--report-every", default=100, type=int)

    # hyper-parameter settings

    args = parser.parse_args()

    LOGGER = utils.getLogger("Ensemble")
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

    datasets = dataset.CIFAR10(args.data_dir)
    testloader = data.DataLoader(dataset = datasets.splits["test"], \
            batch_size = args.batch_size, num_workers = args.num_workers, shuffle = False)

    # Construct the network
    net1 = torch.load(args.load_net1)
    net1 = net1.to(DEVICE)
    LOGGER.info("Load checkpoint from {}".format(args.load_net1))
    num_params = utils.get_params(net1)
    LOGGER.info("Parameter size: {:.5f}M".format(num_params / 1.e6))
    net2 = torch.load(args.load_net2)
    net2 = net2.to(DEVICE)
    LOGGER.info("Load checkpoint from {}".format(args.load_net2))
    num_params = utils.get_params(net2)
    LOGGER.info("Parameter size: {:.5f}M".format(num_params / 1.e6))
    #net3 = torch.load(args.load_net3)
   # net3 = net3.to(DEVICE)
    #LOGGER.info("Load checkpoint from {}".format(args.load_net3))
    net = [net1, net2]

    out_res = []
    out_acc = []
    out_acc_top5 = []

    for i in range(len(net)):
        scores, acc, acc_top5 = valid(net[i], testloader, DEVICE,  args.report_every, LOGGER)
        LOGGER.info("Test epoch: Acc. Top-1 {:.3f}%; Top-5 {:.3f}%".format(acc, acc_top5))
        out_res.append(scores)
        out_acc.append(acc)
        out_acc_top5.append(acc_top5)

    ave_res = (out_res[0]+out_res[1]+out_res[2])/3
    all_prediction = torch.max(ave_res, 1)[1]
    labels_all = torch.tensor([])
    labels_all = labels_all.to(DEVICE)

    for step, (inputs, labels) in enumerate(testloader):
        labels = labels.to(DEVICE)
        labels_all = torch.cat((labels_all, labels), 0)

    ensemble_acc = torch.sum(all_prediction == labels_all) * 1.00 / 10000
    print(out_acc)
    print(out_acc_top5)
    print(ensemble_acc)
    LOGGER.info("Test epoch: Acc. Top-1 {:.3f}%; Top-5 {:.3f}%; Ensemble acc {:.3f}%".format(acc, acc_top5, ensemble_acc))

if __name__ == "__main__":
    main()










