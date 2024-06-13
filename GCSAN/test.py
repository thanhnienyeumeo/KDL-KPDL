import numpy as np
import torch
import datetime
from utils import build_graph, Data, split_validation
import pickle
from model import * 
import argparse
import time

def validate(model, test_data):   
    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit, mrr = [], []
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        targets, scores = forward(model, i, test_data)
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    print('end predicting: ', datetime.datetime.now())
    return hit, mrr

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='diginetica', help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample')
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
opt = parser.parse_args()




# train_data = pickle.load(open('../datasets/' + '/train.pkl', 'rb'))
# train_data = Data(train_data, shuffle=True, opt = opt)
test_data = pickle.load(open('datasets' + '/test.pkl', 'rb'))
test_data = Data(test_data, shuffle=False, opt = opt)
n_node = 22055



model = trans_to_cuda(SessionGraph(opt, n_node, test_data.len_max))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load(opt.saved_data + '/best_mrr.pt', map_location=device)

recall, mrr = validate(model, test_data)
print('recall@20: ', recall)
print('mrr@20: ', mrr)