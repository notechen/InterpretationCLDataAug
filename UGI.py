import argparse
import gc
import cv2
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler
from models.resnet_multi_bn import resnet18, proj_head
from utils import *
import torchvision.transforms as transforms
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

import numpy as np

from data.cifar10 import CustomCIFAR10, CustomCIFAR100, CustomSTL10, CustomTinyImageNet
from optimizer.lars import LARS

parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')
parser.add_argument('--experiment', type=str, default='UAT_mask_tiny_imagenet', help='location for saving trained models')
parser.add_argument('--data', type=str, default='dataset', help='location of the data')
parser.add_argument('--dataset', type=str, default='tiny_imagenet', help='which dataset to be used, (cifar10 or cifar100 or STL_10 or tiny_imagenet)')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
parser.add_argument('--print_freq', default=50, type=int, help='print frequency')
parser.add_argument('--checkpoint', default='', type=str, help='saving pretrained model')
parser.add_argument('--resume', action='store_true', help='if resume training')
parser.add_argument('--optimizer', default='lars', type=str, help='optimizer type')
parser.add_argument('--lr', default=5.0, type=float, help='optimizer lr')
parser.add_argument('--scheduler', default='cosine', type=str, help='lr scheduler type')
parser.add_argument('--ACL_DS', action='store_true', help='if specified, use pgd dual mode,(cal both adversarial and clean)')
parser.add_argument('--twoLayerProj', action='store_true', help='if specified, use two layers linear head for simclr proj head')
parser.add_argument('--pgd_iter', default=5, type=int, help='how many iterations employed to attack the model')
parser.add_argument('--seed', type=int, default=1, help='random seed')


def cosine_annealing(step, total_steps, lr_max, lr_min, warmup_steps=0):
    assert warmup_steps >= 0

    if step < warmup_steps:
        lr = lr_max * step / warmup_steps
    else:
        lr = lr_min + (lr_max - lr_min) * 0.5 * (
                    1 + np.cos((step - warmup_steps) / (total_steps - warmup_steps) * np.pi))

    return lr


def main():
    global args
    args = parser.parse_args()

    assert args.dataset in ['cifar100', 'cifar10', 'STL_10', 'tiny_imagenet']

    save_dir = os.path.join('checkpoints', args.experiment)
    if os.path.exists(save_dir) is not True:
        os.system("mkdir -p {}".format(save_dir))

    log = logger(path=save_dir)
    log.info(str(args))
    setup_seed(args.seed)

    # different attack corresponding to different bn settings
    bn_names = ['normal', ]

    # define model
    model = resnet18(pretrained=False, bn_names=bn_names)

    ch = model.fc.in_features
    model.fc = proj_head(ch, bn_names=bn_names, twoLayerProj=args.twoLayerProj)
    model.cuda()
    cudnn.benchmark = True

    strength = 1.0
    pic_size = 32
    if args.dataset == 'STL_10':
        pic_size = 64
    elif args.dataset == 'tiny_imagenet':
        pic_size = 64
    rnd_color_jitter = transforms.RandomApply([transforms.ColorJitter(0.4 * strength, 0.4 * strength, 0.4 * strength, 0.1 * strength)], p=0.8 * strength)
    rnd_gray = transforms.RandomGrayscale(p=0.2 * strength)
    tfs_train = transforms.Compose([
        transforms.RandomResizedCrop(pic_size, scale=(1.0 - 0.5 * strength, 1.0), interpolation=3),
        transforms.RandomHorizontalFlip(),
        rnd_color_jitter,
        rnd_gray,
        transforms.ToTensor(),
        # transforms.RandomErasing(p=1, scale=(0.1, 0.3), ratio=(0.8, 1.0), value=1, inplace=True)
    ])
    tfs_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    # dataset process
    if args.dataset == 'cifar10':
        train_datasets = CustomCIFAR10(root=args.data, train=True, transform=tfs_train, download=True)
        val_train_datasets = datasets.CIFAR10(root=args.data, train=True, transform=tfs_test, download=True)
        test_datasets = datasets.CIFAR10(root=args.data, train=False, transform=tfs_test, download=True)
        num_classes = 10
    elif args.dataset == 'cifar100':
        train_datasets = CustomCIFAR100(root=args.data, train=True, transform=tfs_train, download=True)
        val_train_datasets = datasets.CIFAR100(root=args.data, train=True, transform=tfs_test, download=True)
        test_datasets = datasets.CIFAR100(root=args.data, train=False, transform=tfs_test, download=True)
        num_classes = 100
    elif args.dataset == 'STL_10':
        train_datasets = CustomSTL10(root=args.data, split='train+unlabeled', transform=tfs_train, download=True)
        val_train_datasets = datasets.STL10(root=args.data, split='train', transform=tfs_test, download=True)
        test_datasets = datasets.STL10(root=args.data, split='test', transform=tfs_test, download=True)
        num_classes = 10
    elif args.dataset == 'tiny_imagenet':
        train_datasets = CustomTinyImageNet(root=os.path.join(args.data, 'tiny-imagenet-200/train'), transform=tfs_train)
        val_train_datasets = datasets.ImageFolder(root=os.path.join(args.data, 'tiny-imagenet-200/train'), transform=tfs_test)
        test_datasets = datasets.ImageFolder(root=os.path.join(args.data, 'tiny-imagenet-200/val'), transform=tfs_test)
        num_classes = 200
    else:
        print("unknow dataset")
        assert False

    train_loader = torch.utils.data.DataLoader(
        train_datasets,
        num_workers=4,
        batch_size=args.batch_size,
        shuffle=True)

    val_train_loader = torch.utils.data.DataLoader(
        val_train_datasets,
        num_workers=4,
        batch_size=args.batch_size,
        shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        test_datasets,
        num_workers=4,
        batch_size=args.batch_size)

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'lars':
        optimizer = LARS(model.parameters(), lr=args.lr, weight_decay=1e-6)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-6, momentum=0.9)
    else:
        print("no defined optimizer")
        assert False

    if args.scheduler == 'constant':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.epochs * len(train_loader) * 10, ],
                                                         gamma=1)
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_annealing(step,
                                                    args.epochs * len(train_loader),
                                                    1,  # since lr_lambda computes multiplicative factor
                                                    1e-6 / args.lr,
                                                    warmup_steps=10 * len(train_loader))
        )
    else:
        print("unknown schduler: {}".format(args.scheduler))
        assert False

    start_epoch = 1
    if args.checkpoint != '':
        checkpoint = torch.load(args.checkpoint)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)

    if args.resume:
        if args.checkpoint == '':
            checkpoint = torch.load(os.path.join(save_dir, 'model.pt'))
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)

        if 'epoch' in checkpoint and 'optim' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            optimizer.load_state_dict(checkpoint['optim'])
            for i in range((start_epoch - 1) * len(train_loader)):
                scheduler.step()
            log.info("resume the checkpoint {} from epoch {}".format(args.checkpoint, checkpoint['epoch']))
        else:
            log.info("cannot resume since lack of files")
            assert False

    for epoch in range(start_epoch, args.epochs + 1):

        log.info("current lr is {}".format(optimizer.state_dict()['param_groups'][0]['lr']))
        train(train_loader, model, optimizer, scheduler, epoch, log, num_classes=num_classes)

        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optim': optimizer.state_dict(),
        }, filename=os.path.join(save_dir, 'model.pt'))

        if epoch % 10 == 0 and epoch > 0:
            # evaluate acc
            acc, tacc = validate(val_train_loader, test_loader, model, log, num_classes=num_classes)
            log.info('train_accuracy {acc:.3f}'
                     .format(acc=acc))
            log.info('test_accuracy {tacc:.3f}'
                     .format(tacc=tacc))

        if epoch % 100 == 0:
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optim': optimizer.state_dict(),
                'acc': acc,
                'tacc': tacc,
            }, filename=os.path.join(save_dir, 'model_{}.pt'.format(epoch)))


def train(train_loader, model, optimizer, scheduler, epoch, log, num_classes):

    losses = AverageMeter()
    losses.reset()
    data_time_meter = AverageMeter()
    train_time_meter = AverageMeter()

    end = time.time()
    for i, (inputs) in enumerate(train_loader):
        data_time = time.time() - end
        data_time_meter.update(data_time)

        scheduler.step()

        d = inputs.size()
        # print("inputs origin shape is {}".format(d))
        inputs = inputs.view(d[0]*2, d[2], d[3], d[4]).cuda()

        model.zero_grad()
        # p = model.eval()(inputs, 'normal')
        inputs_adv = PGD_contrastive(model, inputs, num_classes, iters=args.pgd_iter)  # get adversarial input
        # inputs_adv = torch.nn.Parameter(inputs_adv)
        # new_p = model.eval()(inputs_adv.cuda(), 'normal')
            
        # kl = F.kl_div(F.log_softmax(new_p, dim=-1), F.softmax(p, dim=-1), reduction='sum')
        # kl.backward()
        # CAM = torch.sum(inputs_adv.grad, dim=1).cpu().numpy()
        CAM = torch.sum(torch.abs(inputs_adv - inputs), dim=1).cpu().numpy()
        # inputs_adv.grad = None
        #print(CAM)
        #print(inputs.grad.shape)  # (1024, 3, 32, 32)
        #print(inputs.grad)

        # erase the input under the guidance of UGI
        for j in range(d[0]*2):
            # CIFAR-10/100 sigma=9, STL-10 and Tiny ImageNet sigma=18
            CAM[j] = gaussian_filter(CAM[j],sigma=18)
            CAM[j] = CAM[j] - np.min(CAM[j])
            CAM[j] = CAM[j] / np.max(CAM[j])

            heatmap = cv2.applyColorMap(np.uint8(255 * CAM[j]), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            tranheat = heatmap.transpose(2, 0, 1)
            x, y = np.where(((tranheat[0] == 0) & (tranheat[1] == 0)))
            p_CAM = np.float32(inputs[j].detach().cpu())
            p_CAM[:, x, y] = 1
            inputs[j] = torch.from_numpy(p_CAM)

        model.zero_grad()
        features = model.train()(inputs.cuda(), 'normal')
        loss = nt_xent(features)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(float(loss.detach().cpu()), inputs.shape[0])

        train_time = time.time() - end
        end = time.time()
        train_time_meter.update(train_time)

        gc.collect()
        torch.cuda.empty_cache()
        if i % args.print_freq == 0:
            log.info('Epoch: [{0}][{1}/{2}]\t'
                     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                     'data_time: {data_time.val:.2f}\t'
                     'iter_train_time: {train_time.avg:.2f}\t'.format(
                          epoch, i, len(train_loader), loss=losses,
                          data_time=data_time_meter, train_time=train_time_meter))

    return losses.avg


def validate(train_loader, val_loader, model, log, num_classes=10):
    """
    Run evaluation
    """
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    train_time_meter = AverageMeter()
    losses = AverageMeter()
    losses.reset()
    end = time.time()

    # train a fc on the representation
    for param in model.parameters():
        param.requires_grad = False

    previous_fc = model.fc
    ch = model.fc.in_features
    model.fc = nn.Linear(ch, num_classes)
    model.cuda()

    epochs_max = 100
    lr = 0.1

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0, momentum=0.9, nesterov=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(step,
                                                epochs_max * len(train_loader),
                                                1,  # since lr_lambda computes multiplicative factor
                                                1e-6 / lr,
                                                warmup_steps=0)
    )

    for epoch in range(epochs_max):
        log.info("current lr is {}".format(optimizer.state_dict()['param_groups'][0]['lr']))

        for i, (sample) in enumerate(train_loader):
            scheduler.step()

            x, y = sample[0].cuda(), sample[1].cuda()
            p = model.eval()(x, 'normal')
            loss = criterion(p, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.update(float(loss.detach().cpu()))

            train_time = time.time() - end
            end = time.time()
            train_time_meter.update(train_time)

        log.info('Test epoch: ({0})\t'
                 'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                 'train_time: {train_time.avg:.2f}\t'.format(
            epoch, loss=losses, train_time=train_time_meter))

    acc = []
    for loader in [train_loader, val_loader]:
        losses = AverageMeter()
        losses.reset()
        top1 = AverageMeter()

        for i, (inputs, targets) in enumerate(loader):
            inputs = inputs.cuda()
            targets = targets.cuda()

            # compute output
            with torch.no_grad():
                outputs = model.eval()(inputs, 'normal')
                loss = criterion(outputs, targets)

            outputs = outputs.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(outputs.data, targets)[0]
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))

            if i % args.print_freq == 0:
                log.info('Test: [{0}/{1}]\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Accuracy {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(loader), loss=losses, top1=top1))

        acc.append(top1.avg)

    # recover every thing
    model.fc = previous_fc
    model.cuda()
    for param in model.parameters():
        param.requires_grad = True

    return acc


def save_checkpoint(state, filename='weight.pt'):
    """
    Save the training model
    """
    torch.save(state, filename)


def PGD_contrastive(model, inputs, num_classes,  eps=8. / 255., alpha=2. / 255., iters=10):

    delta = torch.zeros_like(inputs)
    delta = torch.nn.Parameter(delta)

    for i in range(iters):
        features = model.eval()(inputs + delta, 'normal')

        model.zero_grad()
        #
        if i == 0:
            index = np.argsort(features.cpu().data.numpy(), axis=1)[:, -2]  # the second number index
            index_max = np.argmax(features.cpu().data.numpy(), axis=1)

        one_hot = np.zeros((features.size()[0], features.size()[-1]), dtype=np.float32)
        one_hot_max = np.zeros((features.size()[0], features.size()[-1]), dtype=np.float32)
        # print('one_hot', one_hot.shape)

        for i in range(one_hot.shape[0]):
            one_hot[i, index[i]] = 1
            one_hot_max[i, index_max[i]] = 1
        # print(one_hot.shape)
        one_hot = torch.Tensor(torch.from_numpy(one_hot))
        one_hot.requires_grad = True
        one_hot_max = torch.Tensor(torch.from_numpy(one_hot_max))
        one_hot_max.requires_grad = True
        loss = torch.sum(one_hot_max * features.cpu() - one_hot * features.cpu()).cuda()

        loss.backward()

        delta.data = delta.data - alpha * delta.grad.sign()
        delta.grad = None
        delta.data = torch.clamp(delta.data, min=-eps, max=eps)
        delta.data = torch.clamp(inputs + delta.data, min=0, max=1) - inputs


    return (inputs + delta).detach()


if __name__ == '__main__':
    main()


