from NARM.train import *
import torch
import argparse
opt = {
    'dataset_path':'datasets',
    'batch_size': 75,
    'hidden_size': 120,
    'embed_dim': 50,
    'epoch': 15,
    'lr':0.001,
    'lr_dc':0.1,
    'lr_dc_step':3,
    'test':None,
    'topk':20,
    'valid_portion':0.1
}

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', default = opt['dataset_path'])
parser.add_argument('--batch_size', type = int, default = opt['batch_size'])
parser.add_argument('--hidden_size', type = int, default = opt['hidden_size'])
parser.add_argument('--embed_dim', type = int, default = opt['embed_dim'])
parser.add_argument('--epoch', type = int, default = opt['epoch'])
parser.add_argument('--lr', type = float, default = opt['lr'])
parser.add_argument('--lr_dc', type = float, default = opt['lr_dc'])
parser.add_argument('--lr_dc_step', type = int, default = opt['lr_dc_step'])
parser.add_argument('--test', type = str, default = opt['test'])
parser.add_argument('--topk', type = int, default = opt['topk'])
parser.add_argument('--valid_portion', type = float, default = opt['valid_portion'])
parser.add_argument('--saved_data', type = str, default = 'NARM')

opt = parser.parse_args()
n_items = 22055


model = NARM(hidden_size = opt.hidden_size, n_items = n_items, embedding_dim = opt.embed_dim, n_layers=2, dropout=0.25).to(device)

model = torch.load(opt.saved_data + '/best_mrr.pt')
test_data = pickle.load(open('datasets/test.pkl', 'rb'))
test_data = RecSysDataset(test_data)
test_loader = DataLoader(test_data, batch_size = opt.batch_size, shuffle = False, collate_fn = collate_fn)

hit, mrr = validate(test_loader, model)
print('recall@20: ', hit*100)
print('mrr@20: ', mrr*100)
