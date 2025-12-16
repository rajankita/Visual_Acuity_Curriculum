# original code: https://github.com/dyhan0920/PyramidNet-PyTorch/blob/master/train.py

import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from nets.allcnn import AllConvNet
import nets.resnet as RN
from nets.preactnets import PreActResNet18
import utils
import numpy as np
from PIL import Image
import random
import ruamel.yaml as yaml

import warnings
warnings.filterwarnings("ignore")

from utils_train_test import validate, validate_corrupted, AverageMeter, accuracy

seed = 42    
os.environ['PYTHONHASHSEED'] = str(seed)
# Torch RNG
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# Python RNG
np.random.seed(seed)
random.seed(seed)

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Blur curriculum training, continuous')
parser.add_argument('--net_type', default='preactresnet18', type=str,
                    help='networktype: allconv, preactresnet18, and resnet')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--depth', default=32, type=int,
                    help='depth of the network (default: 32)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='to use basicblock for CIFAR datasets (default: bottleneck)')
parser.add_argument('--dataset', dest='dataset', default='imagenet', type=str,
                    help='dataset (options: cifar10, cifar100, and imagenet)')
parser.add_argument('--no-verbose', dest='verbose', action='store_false',
                    help='to print the status at every iteration')
parser.add_argument('--alpha', default=300, type=float,
                    help='number of new channel increases per depth (default: 300)')
parser.add_argument('--expname', default='TEST', type=str,
                    help='name of experiment')
parser.add_argument('--beta', default=0, type=float,
                    help='hyperparameter beta')
parser.add_argument('--cutmix_prob', default=0, type=float,
                    help='cutmix probability')
parser.add_argument('--cfg', type=str, default='')

# curriculum related
parser.add_argument('--sigma_max', type=float,
                    help = 'sigma for curriculum segments')
parser.add_argument('--alpha_init', type=float, default=1.0, 
                    help = 'initial blur probability')
parser.add_argument('--blur_decay_rate', type=float, default=0.1,
                    help = 'rate at which proportion of blurred samples is decreased')

parser.set_defaults(bottleneck=True)
parser.set_defaults(verbose=True)

best_acc1 = 0
best_acc5 = 0


def over_write_args_from_file(args, yml):
    """
    overwrite arguments acocrding to config file
    """
    if yml == '':
        return
    with open(yml, 'r', encoding='utf-8') as f:
        dic = yaml.load(f.read(), Loader=yaml.Loader)
        for k in dic:
            setattr(args, k, dic[k])


def load_dataset(dataset_name, data_dir, batch_size, workers):
    
    if dataset_name.startswith('cifar'):
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        if dataset_name == 'cifar100':
            train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100(data_dir, train=True, download=True, transform=transform_train),
                batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
            val_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100(data_dir, train=False, transform=transform_test),
                batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
            numberofclass = 100
        elif dataset_name == 'cifar10':
            train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_train),
                batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
            val_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10(data_dir, train=False, transform=transform_test),
                batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
            numberofclass = 10
        else:
            raise Exception('unknown dataset: {}'.format(dataset_name))

    elif dataset_name == 'imagenet':
        traindir = os.path.join(os.path.join(data_dir, 'train'))
        valdir = os.path.join(os.path.join(data_dir, 'val'))
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        jittering = utils.ColorJitter(brightness=0.4, contrast=0.4,
                                      saturation=0.4)
        lighting = utils.Lighting(alphastd=0.1,
                                  eigval=[0.2175, 0.0188, 0.0045],
                                  eigvec=[[-0.5675, 0.7192, 0.4009],
                                          [-0.5808, -0.0045, -0.8140],
                                          [-0.5836, -0.6948, 0.4203]])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                jittering,
                lighting,
                normalize,
            ]))

        train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
            num_workers=workers, pin_memory=True, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=True)
        numberofclass = 1000


    else:
        raise Exception('unknown dataset: {}'.format(dataset_name))
    
    return train_loader, val_loader, numberofclass, normalize


               
def main():

    # torch.manual_seed(1)
    # np.random.seed(1)

    global args, best_acc1, best_acc5
    args = parser.parse_args()
    over_write_args_from_file(args, args.cfg)
    print(args)

    # Load data
    train_loader, val_loader, numberofclass, normalize = load_dataset(args.dataset, args.data_dir, args.batch_size, args.workers)

    # Define model architecture
    print("=> creating model '{}'".format(args.net_type))
    if args.net_type == 'resnet':
        model = RN.ResNet(args.dataset, args.depth, numberofclass, args.bottleneck)
    elif args.net_type == 'allconv':
        model = AllConvNet(numberofclass)
    elif args.net_type == 'preactresnet18':
        model = PreActResNet18(numberofclass, model_width=args.model_width, cifar_norm=True)
    else:
        raise Exception('unknown network architecture: {}'.format(args.net_type))

    model = torch.nn.DataParallel(model).cuda()

    # print(model)
    print('the number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=True)

    my_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[50, 100], last_epoch=-1)

    cudnn.benchmark = True

    # set path for saving training logs
    out_dir = os.path.join(f'runs_2025/{args.dataset}/{args.net_type}/clewr_cont1/alpha_{args.alpha_init}_sigma_{args.sigma_max}_decay_{args.blur_decay_rate}_{args.expname}/')
    print('output directory: ', out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    log_path = os.path.join(out_dir, 'training_log.csv')
    csv_filepath = os.path.join(out_dir, 'corruptions.csv')
    with open(log_path, 'w') as f:
        f.write(str(args))
        f.write('epoch,time(s),train_loss,test_loss,test_acc(%)\n')


    for epoch in range(0, args.epochs):

        # train for one epoch
        train_loss, epoch_time = train(train_loader, model, criterion, optimizer, epoch, args. alpha_init, args.sigma_max, args.blur_decay_rate)

        # evaluate on validation set
        acc1, acc5, val_loss = validate(val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = acc1 >= best_acc1
        # print(f'best_acc: {best_acc1}, acc: {acc1}, is_best: {is_best}')
        best_acc1 = max(acc1, best_acc1)
        if is_best:
            best_acc5 = acc5

        with open(log_path, 'a') as f:
            f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' % (
                (epoch + 1),
                epoch_time,
                train_loss,
                val_loss,
                acc1,
            ))

        print(f'Val accuracy, current = {acc1}, best = {best_acc1} \n')        
            
        save_checkpoint({
            'epoch': epoch,
            'arch': args.net_type,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'best_acc5': best_acc5,
            'optimizer': optimizer.state_dict(),
        }, is_best, out_dir)

        my_lr_scheduler.step()

    print('Best accuracy (top-1 and 5 acc):', best_acc1, best_acc5)

    # Load the best model
    best_model_path = os.path.join(out_dir, 'model_best.pth.tar')
    if os.path.isfile(best_model_path):
        print("=> loading checkpoint '{}'".format(best_model_path))
        torch.load(best_model_path)
        checkpoint = torch.load(best_model_path)
        try:
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            print('Regular model')
        except:
            print('CBS model')
            model.load_state_dict(checkpoint, strict=False)
        print("=> loaded checkpoint '{}'".format(best_model_path))


class BlurMulti(object):
    """Blur the image in a sample.
    """

    def __init__(self, epoch, alpha_init, decay_rate, sigma_max):
        self.alpha = alpha_init * np.exp(-decay_rate*epoch)
        self.sigmas = [sigma_max, 0]
        self.probs = [self.alpha, 1.-self.alpha]

    def __call__(self, sample):
        
        # pick a blur level
        sigma = np.random.choice(self.sigmas, size=1, p=self.probs)[0]
        kernel = int(6*sigma+1)
        sigma = sigma.astype('float')

        # apply blur
        if sigma > 0:
            sample = transforms.functional.gaussian_blur(sample, kernel, sigma) 

        return sample


def train(train_loader, model, criterion, optimizer, epoch, alpha_init, sigma_max, decay_rate):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    # define blur transform
    blur = BlurMulti(epoch, alpha_init, decay_rate, sigma_max)
    # composed = transforms.Compose(blur)
                                
    end = time.time()
    current_LR = get_learning_rate(optimizer)[0]
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        # data_time.update(time.time() - end)

        input = input.cuda()
        target = target.cuda()

        # blur input images
        input = blur(input)

        r = np.random.rand(1)
        if args.beta > 0 and r < args.cutmix_prob:

            # generate mixed sample
            lam = np.random.beta(args.beta, args.beta)
            rand_index = torch.randperm(input.size()[0]).cuda()
            target_a = target
            target_b = target[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
            
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            
            # compute output
            output = model(input)
            loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
        else:
            # compute output
            output = model(input)
            loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(acc1.item(), input.size(0))
        top5.update(acc5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0 and args.verbose == True:
            print('Epoch: [{0}/{1}][{2}/{3}]\t'
                  'LR: {LR:.6f}\t'
                  'Blur prob: {alpha:.6f}\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-acc {top1.val:.4f} ({top1.avg:.4f})\t'.format(
                epoch, args.epochs, i, len(train_loader), alpha=blur.alpha, LR=current_LR, loss=losses, top1=top1))

    # measure elapsed time
    epoch_time = time.time() - end

    print('* Epoch: [{0}/{1}]\t Time {2}\t Top 1-acc {top1.avg:.3f}  \t Train Loss {loss.avg:.3f}'.format(
        epoch, args.epochs, epoch_time, top1=top1, loss=losses))

    # save blurred image for viewing
    # print(input.cpu().shape)
    # img = (input.detach().cpu().numpy())[0]
    # img = np.swapaxes(img, 0, 2)
    # print(img.shape)
    # pilimg = Image.fromarray(np.uint8(img*255))
    # pilimg.save(f'runs/{args.expname}/inp_ep{epoch}.png')

    return losses.avg, epoch_time


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth.tar'):
    filename = save_dir + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, save_dir + 'model_best.pth.tar')


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if args.dataset.startswith('cifar'):
        lr = args.lr * (0.1 ** (epoch // (args.epochs * 0.5))) * (0.1 ** (epoch // (args.epochs * 0.75)))
    elif args.dataset == ('imagenet'):
        if args.epochs == 300:
            lr = args.lr * (0.1 ** (epoch // 75))
        else:
            lr = args.lr * (0.1 ** (epoch // 30))
            
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr



if __name__ == '__main__':
    main()
