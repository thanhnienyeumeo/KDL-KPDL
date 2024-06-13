#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import argparse
import pickle
import time
from utils import build_graph, Data, split_validation
from model import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='yoochoose1_64', help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample')
parser.add_argument('--batchSize', type=int, default=75, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=120, help='hidden state size')
parser.add_argument('--epoch', type=int, default=15, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop ')
parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
parser.add_argument('--dynamic', type=bool, default=False)
parser.add_argument('--saved_data', type = str, default = 'GCSAN')
parser.add_argument('--mini', action = 'store_true', help = 'use the mini dataset to test the code')
parser.add_argument('--model',action = 'store_true', help = 'use the model to test the code')

opt = parser.parse_args()
print(opt)


def main():
    
    train_data = pickle.load(open('datasets'  + '/train.pkl', 'rb'))
    if opt.mini:
        train_data = [train_data[0][:200], train_data[1][:200]] # for testing the code
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open('datasets'  + '/test.pkl', 'rb'))
    # all_train_seq = pickle.load(open('../datasets/' + opt.dataset + '/all_train_seq.txt', 'rb'))
    # g = build_graph(all_train_seq)
    train_data = Data(train_data, shuffle=True, opt=opt)
    test_data = Data(test_data, shuffle=False, opt=opt)
    # del all_train_seq, g
    if opt.dataset == 'diginetica':
        n_node = 43098
    elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
        n_node = 22055
    elif opt.dataset == 'diginetica_users':
        n_node = 57070
    else:
        n_node = 310

    print("test=_data len", test_data.len_max)

    # model = trans_to_cuda(SessionGraph(opt, n_node, test_data.len_max))
    model = trans_to_cuda(SessionGraph(opt, n_node, 73))

    start = time.time()
    best_result = [0, 0]
    best_epoch = [0, 0]
    bad_counter = 0
    start_epoch = 0
    if opt.model:
        ckpt = torch.load(opt.saved_data + '/last_model.pth.tar')
        model.load_state_dict(ckpt['state_dict'])
        model.optimizer.load_state_dict(ckpt['optimizer'])
        model.scheduler.load_state_dict(ckpt['scheduler'])
        best_result = ckpt['best_result']
        best_epoch = ckpt['best_epoch']
        start_epoch = ckpt['epoch']
        print('Successfully loaded the model: ', opt.saved_data)
    for epoch in range(start_epoch, opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        hit, mrr = train_test(model, train_data, test_data)
        flag = 0
        bad_counter += 1

        if hit >= best_result[0]:
            best_result[0] = hit
            best_epoch[0] = epoch
            bad_counter = 1
            torch.save(model, opt.saved_data  + "/best_recall.pt")
        if mrr >= best_result[1]:
            best_result[1] = mrr
            best_epoch[1] = epoch
            bad_counter = 1
            torch.save(model, opt.saved_data + "/best_mrr.pt")
        print('Best Result:')
        print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d'% (best_result[0], best_result[1], best_epoch[0], best_epoch[1]))

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
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))


if __name__ == '__main__':
    main()
