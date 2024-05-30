import argparse
import pickle
import time
import sys

from proc_utils import Dataset, split_validation
from model import *
from torch.utils.tensorboard import SummaryWriter
import torch

def str2bool(v):
    return v.lower() in ('true')

import pickle

def _load_file(filename):
  with open(filename, 'rb') as fn:
    data = pickle.load(fn)
  return data

# Default args used for yoochoose-data-64



class yoochoose_data_64():
    dataset = 'yoochoose_data_64'
    batchSize = 75
    hiddenSize = 120
    epoch = 15
    lr = 0.001
    lr_dc = 0.1
    lr_dc_step = 3
    l2 = 1e-5
    step = 1
    patience = 10
    nonhybrid = False
    validation = True
    valid_portion = 0.1
    saved_data = 'TAGNN++'
    mini = False


# Default args used for Yoochoose1_64

class Yoochoose_arg():
    dataset = 'yoochoose1_64'
    batchSize = 75
    hiddenSize = 120
    epoch = 5
    lr = 0.001
    lr_dc = 0.1
    lr_dc_step = 3
    l2 = 1e-5
    step = 1
    patience = 10
    nonhybrid = False
    validation = True
    valid_portion = 0.1


def main(opt):
    model_save_dir = 'saved/'
    writer = SummaryWriter(log_dir='with_pos/logs')

   
    train_data = _load_file('datasets/train.pkl')
    
    if opt.mini:
        train_data = [train_data[0][:200], train_data[1][:200]] # for testing the code
    if opt.validation:
        train_data, valid_data = split_validation(
            train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = _load_file('datasets/test.pkl')
        print('Testing dataset used validation set')

    train_data = Dataset(train_data, shuffle=True)
    test_data = Dataset(test_data, shuffle=False)

    if opt.dataset == 'yoochoose1_64' :
        n_node = 37484
    elif opt.dataset == 'yoochoose_data_64':
        n_node = 22055

    model = to_cuda(Attention_SessionGraph(opt, n_node))
    start = time.time()
    best_result = [0, 0]
    best_epoch = [0, 0]
    bad_counter = 0

    for epoch in range(opt.epoch):
        print('-' * 50)
        print('Epoch: ', epoch)
        hit, mrr = train_test(model, train_data, test_data)
        
        bad_counter += 1 

        writer.add_scalar('epoch/recall', hit, epoch)
        writer.add_scalar('epoch/mrr', mrr, epoch)

        if hit >= best_result[0]:
            best_result[0] = hit
            best_epoch[0] = epoch
            bad_counter = 1
            torch.save(model,opt.saved_data  + "/best_recall.pt")
            
        if mrr >= best_result[1]:
            best_result[1] = mrr
            best_epoch[1] = epoch
            bad_counter = 1
            torch.save(model, opt.saved_data + "/best_mrr.pt")

        print('Best Result:')
        print('\tRecall@20:\t%.4f\tMRR@20:\t%.4f\tEpoch:\t%d,\t%d' %
              (best_result[0], best_result[1], best_epoch[0], best_epoch[1]))

        ckpt_dict = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': model.optimizer.state_dict(),
            'scheduler': model.scheduler.state_dict(),
            'best_result': best_result,
            'best_epoch': best_epoch
        }
        torch.save(ckpt_dict,  opt.saved_data + f'/last_model.pth.tar')

        if bad_counter >= opt.patience:
            break

    print('-' * 50)
    end = time.time()
    print("Running time: %f seconds" % (end - start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='yoochoose_data_64',
                        help='Dataset name: yoochoose_data_64 | yoochoose1_64')
    parser.add_argument('--defaults', type=str2bool,
                        default=True, help='Use default configuration')
    parser.add_argument('--batchSize', type=int,
                        default=75, help='Batch size')
    parser.add_argument('--hiddenSize', type=int,
                        default=120, help='Hidden state dimensions')
    parser.add_argument('--epoch', type=int, default=15,
                        help='The number of epochs to train for')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Set the Learning Rate')
    parser.add_argument('--lr_dc', type=float, default=0.1,
                        help='Set the decay rate for Learning rate')
    parser.add_argument('--lr_dc_step', type=int, default=3,
                        help='Steps for learning rate decay')
    parser.add_argument('--l2', type=float, default=1e-5,
                        help='Assign L2 Penalty')
    parser.add_argument('--patience', type=int, default=10,
                        help='Used for early stopping criterion')
    parser.add_argument('--validation', action='store_true', help='validation')
    parser.add_argument('--valid_portion', type=float, default=0.1,
                        help='Portion of train set to split into val set')
    parser.add_argument('--nonhybrid', type = bool, default = False, help = 'using only global')
    parser.add_argument('--step', type = int, default = 1, help = 'step of GNN')
    parser.add_argument('--saved_data', type = str, default = 'TAGNN++')
    parser.add_argument('--mini', action = 'store_true', help = 'use the mini dataset')
    opt = parser.parse_args()

    if opt.defaults:
        if opt.dataset == 'yoochoose_data_64':
            opt = yoochoose_data_64()
        else:
            opt = Yoochoose_arg()

    else:
        print("Not using the default configuration")

    main(opt)
