import os
import time

import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import csv
import shutil


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

    acc, err = [], []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        # wrong_k = batch_size - correct_k
        acc.append(correct_k.mul_(100.0 / batch_size))
        # err.append(wrong_k.mul_(100.0 / batch_size))

    return acc


def validate(val_loader, model, criterion, print_freq=100, verbose=True, error=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    # model.module.bn.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input = input.cuda()

        output = model(input)
        loss = criterion(output, target)
        # print('target: ', target)
        # print('output: ', torch.argmax(output, dim=-1))
    
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))

        if error == True:
            err1 = 100. - acc1
            err5 = 100. - acc5
            top1.update(err1.item(), input.size(0))
            top5.update(err5.item(), input.size(0))
        else:
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print(i, print_freq)
        if i % print_freq == 0 and verbose == True:
            print('Test (on val set): [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-acc {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-acc {top5.val:.4f} ({top5.avg:.4f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))
        
    return top1.avg, top5.avg, losses.avg


def validate_corrupted(model, criterion, corrupted_data_dir, out_file, val_loader, normalize, batch_size, num_workers):

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
    err1, err5, val_loss = validate(val_loader, model, criterion, verbose=False, error=True)
    writer.writerow(['Clean data', err1])
    print('Clean data, top-1 error = ', err1)

    # Load labels
    labels = np.load(os.path.join(corrupted_data_dir, 'labels.npy'), 
                allow_pickle=True, fix_imports=True, encoding='bytes')
    print('labels: ', labels.shape)

    # Evaluate on corrupted data
    error_rates = []
    for distortion_name in distortions:
        # rate_arr = show_performance(model, distortion_name, corrupted_data_dir)

        corr_data = np.load(os.path.join(corrupted_data_dir, distortion_name+'.npy'), 
                allow_pickle=True, fix_imports=True, encoding='bytes')
        corr_data = np.swapaxes(corr_data, 1, 3)
        corr_data = np.swapaxes(corr_data, 2, 3)
        corr_data = corr_data / 255.

        # print(distortion_name)
        errs = []
        for severity in range(5):
            x_corr = corr_data[10000*(severity):10000*(severity+1), :].astype('float')
            tensor_x = torch.Tensor(x_corr)
            tensor_x_transformed = normalize(tensor_x)
            
            tensor_y = torch.Tensor(labels[10000*(severity):10000*(severity+1)]).long()
            my_dataset = TensorDataset(tensor_x_transformed, tensor_y) # create your datset
            my_dataloader = DataLoader(my_dataset, num_workers=num_workers, 
                            batch_size = batch_size, pin_memory=True) # create your dataloader
            err1, err5, _ = validate(my_dataloader, model, criterion, verbose=False, error=True)
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


def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth.tar'):
    filename = save_dir + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, save_dir + 'model_best.pth.tar')



