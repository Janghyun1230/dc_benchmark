# Codes borrowed from https://github.com/snu-mllab/Efficient-Dataset-Condensation

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time


def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))


def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


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


def random_indices(y, nclass=10, intraclass=False, device='cuda'):
    n = len(y)
    if intraclass:
        index = torch.arange(n).to(device)
        for c in range(nclass):
            index_c = index[y == c]
            if len(index_c) > 0:
                randidx = torch.randperm(len(index_c))
                index[y == c] = index_c[randidx]
    else:
        index = torch.randperm(n).to(device)
    return index


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


def train(args, model, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(),
                          args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[2 * args.epochs // 3, 5 * args.epochs // 6], gamma=0.2)

    cur_epoch, best_acc1, best_acc5, acc1, acc5 = 0, 0, 0, 0, 0
    model = model.cuda()
    print(f"Start {args.epochs} epoch training with {args.mixup} mixup")

    # Start training and validation
    for epoch in range(cur_epoch + 1, args.epochs + 1):
        acc1_tr, _, loss_tr = train_epoch(args,
                                          train_loader,
                                          model,
                                          criterion,
                                          optimizer,
                                          epoch=0,
                                          mixup=args.mixup)

        if epoch % args.epoch_print_freq == 0:
            acc1, acc5, loss_val = validate(args, val_loader, model, criterion, epoch)

            is_best = acc1 > best_acc1
            if is_best:
                best_acc1 = acc1
                best_acc5 = acc5
                print(f'Best accuracy (top-1 and 5): {best_acc1:.1f} {best_acc5:.1f}')

        scheduler.step()
    return best_acc1, acc1


def train_epoch(args,
                train_loader,
                model,
                criterion,
                optimizer,
                epoch=0,
                aug=None,
                mixup='vanilla'):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    end = time.time()
    num_exp = 0
    for i, (input, target) in enumerate(train_loader):
        if train_loader.device == 'cpu':
            input = input.cuda()
            target = target.cuda()

        data_time.update(time.time() - end)

        if aug != None:
            with torch.no_grad():
                input = aug(input)

        r = np.random.rand(1)
        if r < args.mix_p and mixup == 'cut':
            # generate mixed sample
            lam = np.random.beta(args.beta, args.beta)
            rand_index = random_indices(target, nclass=args.nclass)

            target_b = target[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
            ratio = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))

            output = model(input)
            loss = criterion(output, target) * ratio + criterion(output, target_b) * (1. - ratio)
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

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.verbose == True:
            print('Epoch: [{0}/{1}][{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-acc {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-acc {top5.val:.4f} ({top5.avg:.4f})'.format(epoch,
                                                                     args.epochs,
                                                                     i,
                                                                     len(train_loader),
                                                                     batch_time=batch_time,
                                                                     data_time=data_time,
                                                                     loss=losses,
                                                                     top1=top1,
                                                                     top5=top5))

        num_exp += len(target)

    if (epoch % args.epoch_print_freq == 0) and args.verbose:
        print(
            '(Train) [Epoch {0}/{1}] {2} Top1 {top1.avg:.1f}  Top5 {top5.avg:.1f}  Loss {loss.avg:.3f}'
            .format(epoch, args.epochs, get_time(), top1=top1, top5=top5, loss=losses))

    return top1.avg, top5.avg, losses.avg


def validate(args, val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()
        output = model(input)

        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))

        top1.update(acc1.item(), input.size(0))
        top5.update(acc5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.verbose == True:
            print('Test (on val set): [{0}/{1}][{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-acc {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-acc {top5.val:.4f} ({top5.avg:.4f})'.format(epoch,
                                                                     args.epochs,
                                                                     i,
                                                                     len(val_loader),
                                                                     batch_time=batch_time,
                                                                     loss=losses,
                                                                     top1=top1,
                                                                     top5=top5))

    print(
        '(Test ) [Epoch {0}/{1}] {2} Top1 {top1.avg:.1f}  Top5 {top5.avg:.1f}  Loss {loss.avg:.3f}'.
        format(epoch, args.epochs, get_time(), top1=top1, top5=top5, loss=losses))
    return top1.avg, top5.avg, losses.avg
