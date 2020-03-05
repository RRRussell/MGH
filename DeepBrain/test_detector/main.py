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

    save_dir = args.save_dir
    print(save_dir)
    
    # Load the model
    if args.resume:
        print(args.resume)
        checkpoint = torch.load(args.resume)
        net.load_state_dict(checkpoint['state_dict'])
    else:
        return

    n_gpu = setgpu(args.gpu)
    args.n_gpu = n_gpu
    net = net.cuda()
    loss = loss.cuda()
    cudnn.benchmark = False #True
    net = DataParallel(net)
    
    traindatadir = config_training['train_preprocess_result_path']
    valdatadir = config_training['val_preprocess_result_path']
    testdatadir = config_training['test_preprocess_result_path']

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
        testfilelist = valfilelist
    
    import data
    
    # Cut the test CT image into several cubes, 
    # each with a side length of sidelen+2*margin and an overlapping area of 2*margin
    margin = 16
    sidelen = 64
    split_comber = SplitComb(sidelen,config['max_stride'],config['stride'],margin,config['pad_value'])
    dataset = data.DataBowl3Detector(
        testdatadir,
        testfilelist,
        config,
        phase='test',
        split_comber=split_comber)
    test_loader = DataLoader(
        dataset,
        batch_size = 1,
        shuffle = False,
        num_workers = args.workers,
        collate_fn = data.collate,
        pin_memory=False)

    test(test_loader, net, get_pbb, save_dir, config)
                
def test(data_loader, net, get_pbb, save_dir, config):
    isfeat = False
    start_time = time.time()
    net.eval()
    split_comber = data_loader.dataset.split_comber

    # Create a bbox folder for storing the generated pbb file.
    if save_dir is not False:
        save_dir_ = os.path.join(save_dir,'bbox')
        if not os.path.exists(save_dir_):
            os.makedirs(save_dir_)
        else:
            import shutil
            shutil.rmtree(save_dir_)
            os.makedirs(save_dir_)
        print(save_dir_)
    
    # Enumerate each generated cube.
    for i_name, (data, target, coord, nzhw) in enumerate(data_loader):
        s = time.time()
        target = [np.asarray(t, np.float32) for t in target]
        lbb = target[0]
        nzhw = nzhw[0]
        name = data_loader.dataset.filenames[i_name].split('/')[-1].split('_clean')[0]
        print(name)
        data = data[0][0]
        coord = coord[0][0]

        n_per_run = args.n_test
        
        splitlist = range(0,len(data)+1,n_per_run)
        if splitlist[-1]!=len(data):
            splitlist.append(len(data))
        outputlist = []
        featurelist = []
        
        # Take each small cube as a unit and input it into the detection network.
        for i in range(len(splitlist)-1):
            with torch.no_grad():
                inputdata = Variable(data[splitlist[i]:splitlist[i+1]]).cuda()
                inputcoord = Variable(coord[splitlist[i]:splitlist[i+1]]).cuda()
            output = net(inputdata,inputcoord)
            outputlist.append(output.data.cpu().numpy())

        output = np.concatenate(outputlist,0)

        # The predicted output is mapped back to the original CT image.
        output = split_comber.combine(output,nzhw=nzhw)
  
        thresh = args.testthresh # -8 #-3
        
        # Get the original coord.
        pbb = get_pbb(output,thresh,ismask=True,nzhw=nzhw)
        if len(pbb)==0:
            print("skip")
            continue

        # Evaluation.
        tp,fp,fn,_,pbb = acc(pbb,lbb,0.9,0,0.1)
        print("lbb",lbb)
        print("pbb",pbb)
        print("in test:tp,fp,fn",[len(tp),len(fp),len(fn)])
        import sys
        sys.stdout.flush()
        e = time.time()
        
        if save_dir is not False:
            np.save(os.path.join(save_dir_, name+'_pbb.npy'), pbb)
            np.save(os.path.join(save_dir_, name+'_lbb.npy'), lbb)
            
    end_time = time.time()

    print('elapsed time is %3.2f seconds' % (end_time - start_time))
 
if __name__ == '__main__':
    main()