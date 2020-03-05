import argparse
import os
import time
import numpy as np
import data
from importlib import import_module
import shutil
from utils import *
import sys
sys.path.append('../')
from split_combine import SplitComb

import torch
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable

from layers import acc
import ipdb

parser = argparse.ArgumentParser(description='PyTorch DataBowl3 Detector')
parser.add_argument('--model', '-m', metavar='MODEL', default='base',
                    help='model')
parser.add_argument('--config', '-c', default='/workspace/DeepBrain/detector/config_training', type=str)
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--save-freq', default='1', type=int, metavar='S',
                    help='save frequency')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save-dir', default="", type=str, metavar='SAVE',
                    help='directory to save checkpoint (default: none)')
parser.add_argument('--test', default=0, type=int, metavar='TEST',
                    help='1 do test evaluation, 0 not')
parser.add_argument('--testthresh', default=-3, type=float,
                    help='threshod for get pbb')
parser.add_argument('--split', default=8, type=int, metavar='SPLIT',
                    help='In the test phase, split the image to 8 parts')
parser.add_argument('--gpu', default='all', type=str, metavar='N',
                    help='use gpu')
parser.add_argument('--n_test', default=1, type=int, metavar='N',
                    help='number of gpu for test')

def main():
    global args
    args = parser.parse_args()
    config_training = import_module(args.config)

    config_training = config_training.config

    torch.manual_seed(0)
    torch.cuda.set_device(0)

    model = import_module(args.model)
    config, net, loss, get_pbb = model.get_model()

    start_epoch = args.start_epoch
    save_dir = args.save_dir
    print(save_dir)
    
    # Whether to load the model and continue training.
    if args.resume:
        print(args.resume)
        checkpoint = torch.load(args.resume)
        net.load_state_dict(checkpoint['state_dict'])
  
    if start_epoch == 0:
        start_epoch = 1
    if not save_dir:
        exp_id = time.strftime('%Y%m%d-%H%M%S', time.localtime())
        save_dir = os.path.join('results', args.model + '-' + exp_id)
    else:
        save_dir = os.path.join('results',save_dir)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logfile = os.path.join(save_dir,'log')

    n_gpu = setgpu(args.gpu)
    args.n_gpu = n_gpu
    net = net.cuda()
    loss = loss.cuda()
    cudnn.benchmark = False #True
    net = DataParallel(net)

    # Load training file name.
    traindatadir = config_training['train_preprocess_result_path']
    valdatadir = config_training['val_preprocess_result_path']

    trainfilelist = []

    for folder in config_training['train_data_path']:
        for f in os.listdir(folder):
            if f.endswith('label.npy') and f[:-4] not in config_training['black_list']:
                trainfilelist.append(f[:-4].replace('_label',''))
    valfilelist = []
    if valdatadir != traindatadir:
        for folder in config_training['val_data_path']:
            for f in os.listdir(folder):
                if f.endswith('label.npy') and f[:-4] not in config_training['black_list']:
                    valfilelist.append(f[:-4].replace('_label',''))
    else:
    # in this case, 80% data for training and 20% for testing
        case_all = len(trainfilelist)
        case_train = int(0.8*case_all)
        trainfilelist_1 = trainfilelist[0:case_train]
        valfilelist = trainfilelist[case_train:]
    
    import data
    # Making training data_loader
    dataset = data.DataBowl3Detector(
        traindatadir,
        trainfilelist_1,
        config,
        phase = 'train')
    train_loader = DataLoader(
        dataset,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = args.workers,
        pin_memory=True)
    '''
    dataset = data.DataBowl3Detector(
        valdatadir,
        valfilelist,
        config)
        #phase = 'val',
        #split_comber=split_comber)
    
    val_loader = DataLoader(
        dataset,
        batch_size = 1,
        shuffle = False,
        num_workers = args.workers,
        pin_memory=True)
    '''
    
    optimizer = torch.optim.SGD(
        net.parameters(),
        args.lr,
        momentum = 0.9,
        weight_decay = args.weight_decay)
    
    # optimizer = torch.optim.Adam(net.parameters(),args.lr)
    
    def get_lr(epoch):
        if epoch <= args.epochs * 1/3:
            lr = args.lr
        elif epoch <= args.epochs * 2/3:
            lr = 0.1 * args.lr
        elif epoch <= args.epochs * 0.8:
            lr = 0.05 * args.lr
        else:
            lr = 0.01 * args.lr
        return lr
    
    for epoch in range(start_epoch, args.epochs + 1):
        train(train_loader, net, loss, epoch, optimizer, get_lr, args.save_freq, save_dir, get_pbb)
            
def train(data_loader, net, loss, epoch, optimizer, get_lr, save_freq, save_dir, get_pbb):
    start_time = time.time()
    
    net.train()
    lr = get_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    metrics = []

    for i, (data, target, coord, fname) in enumerate(data_loader):
        data = Variable(data.cuda(async = True))
        target = Variable(target.cuda(async = True))
        coord = Variable(coord.cuda(async = True))
        output = net(data, coord)
        target = target.type(torch.cuda.FloatTensor)
        '''
        ('data', (batch_size, 1, 96, 96, 96))
        ('target', (batch_size, 24, 24, 24, 3, 5))
        ('coord', (batch_size, 3, 24, 24, 24))
        ('output', (batch_size, 24, 24, 24, 3, 5))
        Here data is a 96 * 96 * 96 cube cut from the original CT image
        Target is a label that records the predicted probability, xyz coordinates, and diameter of each position in the cube.
        Coord records grid coordinate values.
        Output is the output of the model that records the predicted probability, xyz coordinates, and diameter of each position in the cube.
        '''
        loss_output = loss(output, target)
        optimizer.zero_grad()
        loss_output[0].backward()
        optimizer.step()

        loss_output[0] = loss_output[0].item()
        metrics.append(loss_output)

    if epoch % args.save_freq == 0:            
        state_dict = net.module.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()
        
        # save the model
        torch.save({
            'epoch': epoch,
            'save_dir': save_dir,
            'state_dict': state_dict,
            'args': args},
            os.path.join(save_dir, '%03d.ckpt' % epoch))

    end_time = time.time()

    metrics = np.asarray(metrics, np.float32)
    print('Epoch %03d (lr %.5f)' % (epoch, lr))
    print('Train: loss %2.4f, classify loss %2.4f, regress loss %2.4f, %2.4f, %2.4f, %2.4f' % (
        np.mean(metrics[:, 0]),
        np.mean(metrics[:, 1]),
        np.mean(metrics[:, 2]),
        np.mean(metrics[:, 3]),
        np.mean(metrics[:, 4]),
        np.mean(metrics[:, 5])))
    print
    
if __name__ == '__main__':
    main()
    
'''
def validate(data_loader, net, loss, get_pbb):
    start_time = time.time()
    
    net.eval()

    metrics = []
    for i, (data, target, coord) in enumerate(data_loader):
        print(i)
        data = Variable(data.cuda(async = True), volatile = True)
        target = Variable(target.cuda(async = True), volatile = True)
        coord = Variable(coord.cuda(async = True), volatile = True)
        print(target.shape)
        print(coord.shape)

        output = net(data, coord)
        target = target.type(torch.cuda.FloatTensor)
        loss_output = loss(output, target, train = False)

        loss_output[0] = loss_output[0].item()
        metrics.append(loss_output)    
        
        thresh = args.testthresh # -8 #-3
        print('pbb thresh', thresh)
        pbb,mask = get_pbb(output.cpu().detach().numpy(),thresh,ismask=True)
        tp,fp,fn,_ = acc(pbb,target.cpu().detach().numpy(),0,0.1,0.1)
        print([len(tp),len(fp),len(fn)])
    end_time = time.time()

    metrics = np.asarray(metrics, np.float32)
    #print('Validation: tpr %3.2f, total pos %d, total fp %d, time %3.2f' % (
    #    100.0 * np.sum(metrics[:, 6]) / np.sum(metrics[:, 7]),
    #    np.sum(metrics[:, 7]),
    #    np.sum(metrics[:, 7])-np.sum(metrics[:, 6]),
    #    end_time - start_time))
    print('Validation: loss %2.4f, classify loss %2.4f, regress loss %2.4f, %2.4f, %2.4f, %2.4f' % (
        np.mean(metrics[:, 0]),
        np.mean(metrics[:, 1]),
        np.mean(metrics[:, 2]),
        np.mean(metrics[:, 3]),
        np.mean(metrics[:, 4]),
        np.mean(metrics[:, 5])))
    
    print
    print
'''