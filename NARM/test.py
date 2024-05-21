from ultis import *
import torch

args = {
    'dataset_path':'/content/drive/MyDrive/KDL&KPDL/yoochoose-data/yoochoose-data-64',
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
n_items = 22055
model = NARM(hidden_size = args['hidden_size'], n_items = n_items, embedding_dim = args['embed_dim'], n_layers=2, dropout=0.25).to(device)

model = torch.load('NARM/best_mrr.pt')
test_data = pickle.load(open('datasets/test.pkl', 'rb'))
test_data = RecSysDataset(test_data)
test_loader = DataLoader(test_data, batch_size = args['batch_size'], shuffle = False, collate_fn = collate_fn)

hit, mrr = validate(test_loader, model)
print('recall@20: ', hit)
print('mrr@20: ', mrr)
