import argparse
import os
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import TensorDataset, DataLoader
import nets.resnet as RN
import nets.pyramidnet as PYRM
from nets.allcnn import AllConvNet
import nets.resnet_cifar10 as RN_cifar
from nets.preactnets import PreActResNet18
# from nets.mobilenetv2 import MobileNetV2
import numpy as np
import csv

from torch import nn
from torch.nn import functional as F

import warnings
warnings.filterwarnings("ignore")

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Test on clean and corrupted data')
parser.add_argument('--net_type', default='pyramidnet', type=str,
                    help='networktype: resnet, pyamidnet and allconv')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('--depth', default=32, type=int,
                    help='depth of the network (default: 32)')
parser.add_argument('--model_width', default=64, type=int,
                    help='width of the network (default: 64)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='to use basicblock for CIFAR datasets (default: bottleneck)')
parser.add_argument('--data_dir', dest='data_dir', default='../data', type=str,
                    help='Root data directory')
parser.add_argument('--corrupted_data_dir', dest='corrupted_data_dir', default='../data', type=str,
                    help='Corruption data directory')
parser.add_argument('--dataset', dest='dataset', default='imagenet', type=str,
                    help='dataset (options: cifar10, cifar100, and imagenet)')
parser.add_argument('--alpha', default=300, type=float,
                    help='number of new channel increases per depth (default: 300)')
parser.add_argument('--no-verbose', dest='verbose', action='store_true',
                    help='to print the status at every iteration')
parser.add_argument('--pretrained', default='/set/your/model/path', type=str, metavar='PATH')

parser.set_defaults(bottleneck=True)
parser.set_defaults(verbose=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

best_err1 = 100
best_err5 = 100


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=10):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


def get_calibration(data_loader, model, debug=False, n_bins=None):
    "This function taken from noisymix"
    model.eval()
    mean_conf, mean_acc = [], []
    ece = []

    # New code to compute ECE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # ------------------------------------------------------------------------
    logits_list = []
    labels_list = []
    #mean_conf, mean_acc = [], []

    with torch.no_grad():
        for i, (images, target) in enumerate(data_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)  # TODO: one hot or class number? We want class number
            logits = model(images)
            logits_list.append(logits)
            labels_list.append(target)

    logits = torch.cat(logits_list).cuda()
    labels = torch.cat(labels_list).cuda()

    ece_criterion = _ECELoss(n_bins=15).cuda()
    ece = ece_criterion(logits, labels).item()
    #print('ECE (new implementation):', ece * 100)

    #calibration_dict = _get_calibration(labels, torch.nn.functional.softmax(logits, dim=1), debug=False, num_bins=n_bins)
    #mean_conf = calibration_dict['reliability_diag'][0]
    #mean_acc = calibration_dict['reliability_diag'][1]
    #print(mean_conf.shape)
    
    #calibration_results = {'reliability_diag': (torch.vstack(mean_conf), torch.vstack(mean_acc)), 'ece': ece}
    #calibration_results = {'reliability_diag': ((mean_conf), (mean_acc)), 'ece': ece}

    return ece


def load_dataset(data_dir, corrupted_data_dir):
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
                batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
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

    else:
        raise Exception('unknown dataset: {}'.format(args.dataset))

    return val_loader, numberofclass, normalize, corrupted_data_dir


def main():
    global args, best_err1, best_err5
    args = parser.parse_args()

    # Load data
    val_loader, numberofclass, normalize, corrupted_data_dir = load_dataset(args.data_dir, args.corrupted_data_dir)

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
        model = PreActResNet18(numberofclass, model_width=64, cifar_norm=True)
    elif args.net_type == 'MobileNetV2':
        from torchvision.models import MobileNetV2
        model = MobileNetV2()
    else:
        raise Exception('unknown network architecture: {}'.format(args.net_type))

    # uncomment the following if not testing a CBS model
    model = torch.nn.DataParallel(model).cuda()
    model_path = args.pretrained
    if not os.path.isfile(model_path):
        model_path = os.path.join(args.pretrained, 'model_best.pth.tar')

    # uncomment the following for CBS
    # model = model.cuda()
    # model_path = os.path.join(args.pretrained)

    if os.path.isfile(model_path):
        print("=> loading checkpoint '{}'".format(model_path))
        torch.load(model_path)
        checkpoint = torch.load(model_path)
        # model.load_state_dict(checkpoint['state_dict'], strict=True)

        try:
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            print('Regular model')
            csv_filepath = f'{args.pretrained}/corruptions.csv'
        except:
            print('CBS model')
            model.load_state_dict(checkpoint, strict=False)
            csv_filepath = f'{args.pretrained}_corruptions.csv'
        print("=> loaded checkpoint '{}'".format(model_path))
    else:
        raise Exception("=> no checkpoint found at '{}'".format(model_path))

    # print(model)
    print('the number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    # file to store the results
    # if not os.path.exists(args.results):
        # os.makedirs(args.results)
    # csv_filepath = f'{args.pretrained}/corruptions.csv'
    

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    cudnn.benchmark = True

    # evaluate on validation set
    err1, err5, val_loss = validate(val_loader, model, criterion)

    # evaluate on corrupted data
    if args.dataset.startswith('cifar'):
        validate_corrupted(model, criterion, corrupted_data_dir, csv_filepath, val_loader, normalize)

    print('Accuracy (top-1 and 5 error):', err1, err5)


def validate(val_loader, model, criterion, verbose=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input = input.cuda()

        output = model(input)
        loss = criterion(output, target)
        # print('target: ', target)
        # print('output: ', torch.argmax(output, dim=-1))
    
        # measure accuracy and record loss
        err1, err5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))

        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and verbose == True:
            print('Test (on val set): [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-err {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))

    # print(input.min(), input.max())
    return top1.avg, top5.avg, losses.avg


def validate_corrupted(model, criterion, corrupted_data_dir, out_file, val_loader, normalize):

    distortions = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog', 'brightness',
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
    # 'speckle_noise', 'gaussian_blur', 'spatter', 'saturate'
    ]


    # set up filewriter
    f = (open(out_file, 'w'))
    writer = csv.writer(f)
    writer.writerow(['Corruption', 'mCE', 'CE-1', 'CE-2', 'CE-3', 'CE-4', 'CE-5'])

    # Evaluate on clean data
    err1, err5, val_loss = validate(val_loader, model, criterion)
    writer.writerow(['Clean data', err1])
    print('Clean data, top-1 error = ', err1)

    # Load labels
    labels = np.load(os.path.join(corrupted_data_dir, 'labels.npy'), 
                allow_pickle=True, fix_imports=True, encoding='bytes')
    print('labels: ', labels.shape)

    # Evaluate on corrupted data
    error_rates = []
    ece_rates = []
    for distortion_name in distortions:
        # rate_arr = show_performance(model, distortion_name, corrupted_data_dir)

        corr_data = np.load(os.path.join(corrupted_data_dir, distortion_name+'.npy'), 
                allow_pickle=True, fix_imports=True, encoding='bytes')
        corr_data = np.swapaxes(corr_data, 1, 3)
        corr_data = np.swapaxes(corr_data, 2, 3)
        corr_data = corr_data / 255.
        # print('corr data', corr_data.shape)

        # print(distortion_name)
        errs = []
        eces = []
        for severity in range(5):
            x_corr = corr_data[10000*(severity):10000*(severity+1), :].astype('float')
            tensor_x = torch.Tensor(x_corr)
            tensor_x_transformed = normalize(tensor_x)
            
            tensor_y = torch.Tensor(labels[10000*(severity):10000*(severity+1)]).long()
            my_dataset = TensorDataset(tensor_x_transformed, tensor_y) # create your datset
            my_dataloader = DataLoader(my_dataset, num_workers=args.workers, 
                            batch_size = args.batch_size, pin_memory=True) # create your dataloader
            err1, err5, _ = validate(my_dataloader, model, criterion)
            calibration = get_calibration(my_dataloader, model, debug=False)
            # calibration = err1

            errs.append(err1)
            eces.append(calibration)

        rate_mean = np.mean(errs)
        error_rates.append(rate_mean)
        writer.writerow([distortion_name, rate_mean, errs[0], errs[1], errs[2], errs[3], errs[4]])
        print('Distortion: {:15s}  | CE (unnormalized) (%): {:.2f}'.format(distortion_name, rate_mean))

        ece_mean = np.mean(eces)
        ece_rates.append(ece_mean*100)
        writer.writerow([distortion_name, ece_mean, eces[0], eces[1], eces[2], eces[3], eces[4]])
        print('Calibration: {:15s}  | ECE (%): {:.2f}'.format(distortion_name, ece_mean))

    print('mCE (unnormalized by AlexNet errors) (%): {:.2f}'.format(np.mean(error_rates)))
    writer.writerow(['mCE', np.mean(error_rates)])

    print('corruption ECE (%): {:.2f}'.format(np.mean(ece_rates)))
    writer.writerow(['ECE', np.mean(ece_rates)])

    f.close()


def validate_tin_corrupted(model, criterion, corrupted_data_dir, out_file, val_loader, val_transforms):

    distortions = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog', 'brightness',
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression' ]

    # set up filewriter
    f = (open(out_file, 'w'))
    writer = csv.writer(f)
    writer.writerow(['Corruption', 'mCE', 'CE-1', 'CE-2', 'CE-3', 'CE-4', 'CE-5'])

    # Evaluate on clean data
    err1, err5, val_loss = validate(val_loader, model, criterion)
    writer.writerow(['Clean data', err1])
    print('Clean data, top-1 error = ', err1)

    # Evaluate on corrupted data
    error_rates = []
    for distortion_name in distortions:

        errs = []
        for severity in range(5):

            corr_dataset = datasets.ImageFolder(os.path.join(corrupted_data_dir, 
                            distortion_name, f'{severity+1}'), val_transforms) 
            val_loader = torch.utils.data.DataLoader(corr_dataset, batch_size=args.batch_size, 
                        shuffle=True,  num_workers=args.workers, pin_memory=True)

            err1, err5, _ = validate(val_loader, model, criterion)
            # print(correct, error_top1)
            errs.append(err1)
            print(distortion_name, severity, err1)
            # break

        rate_mean = np.mean(errs)
        error_rates.append(rate_mean)
        writer.writerow([distortion_name, rate_mean, errs[0], errs[1], errs[2], errs[3], errs[4]])
        print('Distortion: {:15s}  | CE (unnormalized) (%): {:.2f}'.format(distortion_name, rate_mean))

    print('mCE (unnormalized by AlexNet errors) (%): {:.2f}'.format(np.mean(error_rates)))
    writer.writerow(['mCE', np.mean(error_rates)])

    f.close()


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
    # print('pred = ', pred)
    # print('target = ', target)

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        wrong_k = batch_size - correct_k
        res.append(wrong_k.mul_(100.0 / batch_size))

    return res


if __name__ == '__main__':
    main()
