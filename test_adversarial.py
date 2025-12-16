import argparse
import os
import shutil
import time
import numpy as np
import random

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from nets.allcnn import AllConvNet
import nets.resnet as RN
import nets.pyramidnet as PYRM
import nets.resnet_cifar10 as RN_cifar
from nets.preactnets import PreActResNet18
from advertorch.attacks import PGDAttack, GradientSignAttack

seed = 42    
os.environ['PYTHONHASHSEED'] = str(seed)
# Torch RNG
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# Python RNG
np.random.seed(seed)
random.seed(seed)
# torch.set_deterministic(True)

import warnings
warnings.filterwarnings("ignore")

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Test on adversarial samples')
parser.add_argument('--net_type', default='pyramidnet', type=str,
                    help='networktype: resnet, pyamidnet and allconv')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--depth', default=32, type=int,
                    help='depth of the network (default: 32)')
parser.add_argument('--model_width', default=64, type=int,
                    help='width of the network (default: 64)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='to use basicblock for CIFAR datasets (default: bottleneck)')
parser.add_argument('--data_dir', dest='data_dir', default='../data', type=str,
                    help='Root data directory')
parser.add_argument('--dataset', dest='dataset', default='imagenet', type=str,
                    help='dataset (options: cifar10, cifar100, and imagenet)')
parser.add_argument('--alpha', default=300, type=float,
                    help='number of new channel increases per depth (default: 300)')
parser.add_argument('--no-verbose', dest='verbose', action='store_false',
                    help='to print the status at every iteration')
parser.add_argument('--pretrained', default='/set/your/model/path', type=str, metavar='PATH')
parser.add_argument('--sigmas', nargs='+', type=float,
                    help = 'sigma for evaluation on blurred images')

parser.set_defaults(bottleneck=True)
parser.set_defaults(verbose=True)

best_acc1 = 100
best_acc5 = 100


def load_dataset(data_dir):
    if args.dataset.startswith('cifar'):
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        if args.dataset == 'cifar100':
            val_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100(data_dir, train=False, transform=transform_test),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            numberofclass = 100
        elif args.dataset == 'cifar10':
            val_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10(data_dir, train=False, transform=transform_test),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            numberofclass = 10
        else:
            raise Exception('unknown dataset: {}'.format(args.dataset))

    elif args.dataset == 'imagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(data_dir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        numberofclass = 1000

    elif args.dataset == 'imagenet-sketch':

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(data_dir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        numberofclass = 1000

    else:
        raise Exception('unknown dataset: {}'.format(args.dataset))

    return val_loader, numberofclass, normalize

def main():
    global args, best_acc1, best_acc5
    args = parser.parse_args()

    # Load data
    val_loader, numberofclass, normalize = load_dataset(args.data_dir)

    # Define model architecture
    print("=> creating model '{}'".format(args.net_type))
    if args.net_type == 'resnet_cifar':
        model = RN_cifar.__dict__[args.net_type + f'{args.depth}']()
    elif args.net_type == 'resnet':
        model = RN.ResNet(args.dataset, args.depth, numberofclass, args.bottleneck)
    elif args.net_type == 'pyramidnet':
        model = PYRM.PyramidNet(args.dataset, args.depth, args.alpha, numberofclass,
                                args.bottleneck)
    elif args.net_type == 'allconv':
        model = AllConvNet(numberofclass)
    elif args.net_type == 'preactresnet18':
        model = PreActResNet18(numberofclass, model_width=args.model_width, cifar_norm=True)
    elif args.net_type == 'MobileNetV2':
        from torchvision.models import MobileNetV2
        model = MobileNetV2()
    else:
        raise Exception('unknown network architecture: {}'.format(args.net_type))

    model = torch.nn.DataParallel(model).cuda()
    # model = model.cuda()

    # Load trained model checkpoint
    model_path = args.pretrained
    if os.path.isfile(model_path):
        print("=> loading checkpoint '{}'".format(model_path))
        torch.load(model_path)
        checkpoint = torch.load(model_path)
        try:
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            print('Regular model')
        except:
            print('CBS model')
            model.load_state_dict(checkpoint, strict=False)
        print("=> loaded checkpoint '{}'".format(model_path))
    else:
        raise Exception("=> no checkpoint found at '{}'".format(model_path))

    
    # if os.path.isfile(args.pretrained):
    #     print("=> loading checkpoint '{}'".format(args.pretrained))
    #     checkpoint = torch.load(args.pretrained)
    #     model.load_state_dict(checkpoint['state_dict'], strict=False)
    #     print("=> loaded checkpoint '{}'".format(args.pretrained))
    # else:
    #     raise Exception("=> no checkpoint found at '{}'".format(args.pretrained))

    print(model)
    print('the number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    cudnn.benchmark = True

    # evaluate on adversarial data
    attack = 'PGD'
    for eps in [float(2/255), float(8/255)]:
        acc1, asr = validate_adv(val_loader, model, criterion, eps, attack)
        print(f'Epsilon: {eps}, acc = {acc1}, err = {100-acc1}, asr = {asr}')

    # print('Accuracy (top-1 and 5 acc):', acc1, acc5)
    # print('Top-1 error: ', 100. - acc1)


class Blur(object):
    """Blur the image in a sample.
    """

    def __init__(self, sigma):
        self.sigma = sigma
        self.kernel = int(self.sigma*6 + 1)

    def __call__(self, sample):
        # apply blur
        if self.sigma > 0:
            sample = transforms.functional.gaussian_blur(sample, self.kernel, self.sigma) 

        return sample


def validate_adv(val_loader, model, criterion, eps, attack):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    # PGD
    if attack == 'PGD':
        adversary = PGDAttack(
            model,
            loss_fn=nn.CrossEntropyLoss(reduction="sum"),
            eps=eps,
            nb_iter=6, eps_iter=0.03, clip_min=0.0, clip_max=1.0,
            targeted=False)
    # FGSM
    elif attack == 'FGSM':
        adversary = GradientSignAttack(
            model,
            loss_fn=nn.CrossEntropyLoss(reduction="sum"),
            eps=eps,
            targeted=False)


    end = time.time()
    total = 0
    correct = 0
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()

        # initial prediction on clean data
        output = model(input)
        _, predicted_clean = torch.max(output.data, 1)

        # untargeted adv attack
        adv_inputs_ori = adversary.perturb(input, target)
        # final prediction on perturbed data
        with torch.no_grad():
            outputs = model(adv_inputs_ori)
            _, predicted_adv = torch.max(outputs.data, 1)
        # test the images for which the initial prediction is correct
        for (gt, cl, adv) in zip(target, predicted_clean, predicted_adv):
            if cl == gt:    
                total += 1
                correct += (adv == gt).sum()

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))

        top1.update(acc1.item(), input.size(0))
        top5.update(acc5.item(), input.size(0))

    asr = 100.0 - 100. * correct.float() / total
    print('Attack success rate: %.2f %%' %asr)
    print(f'{total-correct}/{total}')
    
    return top1.avg, asr


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        # wrong_k = batch_size - correct_k
        # res_acc.append(correct_k.mul_(100.0 / batch_size))
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


if __name__ == '__main__':
    main()
