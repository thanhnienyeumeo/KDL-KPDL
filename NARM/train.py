import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import time
import random

from tqdm import tqdm

from torch.utils.data import DataLoader

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from torch.backends import cudnn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def load_data(root='/content/drive/MyDrive/yoochoose-data-64', valid_portion=0.1, maxlen=19, sort_by_len=False, train_set=None, test_set=None):
    """Load dataset từ root
    root: folder dữ liệu train, trong trường hợp train_set, test_set tồn tại thì không sử dụng train_set và test_set
    valid_portion: tỷ lệ phân chia dữ liệu validation/train
    maxlen: độ dài lớn nhất của sequence
    sort_by_len: có sort theo chiều dài các session trước khi chia hay không?
    train_set: training dataset
    test_set:  test dataset
    """

    # Load the dataset
    if train_set is None and test_set is None:
        path_train_data = os.path.join(root, 'train.pkl')
        path_test_data = os.path.join(root, 'test.pkl')
        with open(path_train_data, 'rb') as f1:
            train_set = pickle.load(f1)

        with open(path_test_data, 'rb') as f2:
            test_set = pickle.load(f2)

    if maxlen:
        new_train_set_x = []
        new_train_set_y = []
        # Lọc dữ liệu sequence đến maxlen
        for x, y in zip(train_set[0], train_set[1]):
            if len(x) < maxlen:
                new_train_set_x.append(x)
                new_train_set_y.append(y)
            else:
                new_train_set_x.append(x[:maxlen])
                new_train_set_y.append(y)
        train_set = (new_train_set_x, new_train_set_y)
        del new_train_set_x, new_train_set_y

        new_test_set_x = []
        new_test_set_y = []
        for xx, yy in zip(test_set[0], test_set[1]):
            if len(xx) < maxlen:
                new_test_set_x.append(xx)
                new_test_set_y.append(yy)
            else:
                new_test_set_x.append(xx[:maxlen])
                new_test_set_y.append(yy)
        test_set = (new_test_set_x, new_test_set_y)
        del new_test_set_x, new_test_set_y

    # phân chia tập train thành train và validation
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    (test_set_x, test_set_y) = test_set

    # Trả về indices thứ tự độ dài của mỗi phần tử trong seq
    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    # Sắp xếp session theo độ dài tăng dần
    if sort_by_len:
        sorted_index = len_argsort(test_set_x)
        test_set_x = [test_set_x[i] for i in sorted_index]
        test_set_y = [test_set_y[i] for i in sorted_index]

        sorted_index = len_argsort(valid_set_x)
        valid_set_x = [valid_set_x[i] for i in sorted_index]
        valid_set_y = [valid_set_y[i] for i in sorted_index]

    train = (train_set_x, train_set_y)
    valid = (valid_set_x, valid_set_y)
    test = (test_set_x, test_set_y)
    return train, valid, test



def collate_fn(data):
    """
    Hàm số này sẽ được sử dụng để pad session về max length
    Args:
      data: batch truyền vào
    return:
      batch data đã được pad length có shape maxlen x batch_size
    """
    # Sort batch theo độ dài của input_sequence từ cao xuống thấp
    data.sort(key=lambda x: len(x[0]), reverse=True)
    lens = [len(sess) for sess, label in data]
    labels = []
    # Padding batch size
    padded_sesss = torch.zeros(len(data), max(lens)).long()
    for i, (sess, label) in enumerate(data):
        padded_sesss[i,:lens[i]] = torch.LongTensor(sess)
        labels.append(label)

    # Transpose dữ liệu từ batch_size x maxlen --> maxlen x batch_size
    padded_sesss = padded_sesss.transpose(0,1)
    return padded_sesss, torch.tensor(labels).long(), lens



class RecSysDataset(Dataset):
    """define the pytorch Dataset class for yoochoose and diginetica datasets.
    """
    def __init__(self, data):
        self.data = data
        print('-'*50)
        print('Dataset info:')
        print('Number of sessions: {}'.format(len(data[0])))
        print('-'*50)

    def __getitem__(self, index):
        session_items = self.data[0][index]
        target_item = self.data[1][index]
        return session_items, target_item

    def __len__(self):
        return len(self.data[0])
    

import math
class NARM(nn.Module):
    def __init__(self, hidden_size, n_items, embedding_dim, n_layers=1, dropout=0.25):
        super(NARM, self).__init__()
        self.hidden_size = hidden_size
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(self.n_items, self.embedding_dim, padding_idx = 0)
        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        # set bidirectional = True for bidirectional
        # https://pytorch.org/docs/stable/nn.html?highlight=gru#torch.nn.GRU to get more information
        self.gru = nn.GRU(input_size = hidden_size, # number of expected feature of input x
                          hidden_size = hidden_size, # number of expected feature of hidden state
                          num_layers = n_layers, # number of GRU layers
                          dropout=(0 if n_layers == 1 else dropout), # dropout probability apply in encoder network
                          bidirectional=True # one or two directions.
                         )
        self.emb_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(self.embedding_dim, self.hidden_size, self.n_layers)
        self.a_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.a_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_t = nn.Linear(self.hidden_size, 1, bias=False)
        self.ct_dropout = nn.Dropout(0.5)
        self.b = nn.Linear(self.embedding_dim, 2 * self.hidden_size, bias=False)
        self.sf = nn.Softmax()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.reset_parameters()
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input_seq, input_lengths, hidden=None):
        """
        input_seq: Batch input_sequence. Shape: max_len x batch_size
        input_lengths: Batch input lengths. Shape: batch_size
        """
        # Step 1: Convert sequence indexes to embeddings
        # shape: (max_length , batch_size , hidden_size)
        embedded = self.embedding(input_seq)
        # Pack padded batch of sequences for RNN module. Padding zero when length less than max_length of input_lengths.
        # shape: (max_length , batch_size , hidden_size)
        packed = pack_padded_sequence(embedded, input_lengths)

        # Step 2: Forward packed through GRU
        # outputs is output of final GRU layer
        # hidden is concatenate of all hidden states corresponding with each time step.
        # outputs shape: (max_length , batch_size , hidden_size x num_directions)
        # hidden shape: (n_layers x num_directions , batch_size , hidden_size)
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding. Revert of pack_padded_sequence
        # outputs shape: (max_length , batch_size , hidden_size x num_directions)
        outputs, length = pad_packed_sequence(outputs)

        # Step 3: Global Encoder & Local Encoder
        # num_directions = 1 -->
        # outputs shape:(max_length , batch_size , hidden_size)
        # hidden shape: (n_layers , batch_size , hidden_size)
        # lấy hidden state tại time step cuối cùng
        ht = hidden[-1]
        # reshape outputs
        outputs = outputs.permute(1, 0, 2) # [batch_size, max_length, hidden_size]
        c_global = ht
        # Flatten outputs thành shape: [batch_size, max_length, hidden_size]
        gru_output_flatten = outputs.contiguous().view(-1, self.hidden_size)
        # Thực hiện một phép chiếu linear projection để tạo các latent variable có shape [batch_size, max_length, hidden_size]
        q1 = self.a_1(gru_output_flatten).view(outputs.size())
        # Thực hiện một phép chiếu linear projection để tạo các latent variable có shape [batch_size, max_length, hidden_size]
        q2 = self.a_2(ht)
        # Ma trận mask đánh dấu vị trí khác 0 trên padding sequence.
        mask = torch.where(input_seq.permute(1, 0) > 0, torch.tensor([1.], device = self.device), torch.tensor([0.], device = self.device)) # batch_size x max_len
        # Điều chỉnh shape
        q2_expand = q2.unsqueeze(1).expand_as(q1) # shape [batch_size, max_len, hidden_size]
        q2_masked = mask.unsqueeze(2).expand_as(q1) * q2_expand # batch_size x max_len x hidden_size
        # Tính trọng số alpha đo lường similarity giữa các hidden state
        alpha = self.v_t(torch.sigmoid(q1 + q2_masked).view(-1, self.hidden_size)).view(mask.size()) # batch_size x max_len
        alpha_exp = alpha.unsqueeze(2).expand_as(outputs) # batch_size x max_len x hidden_size
        # Tính linear combinition của các hidden state
        c_local = torch.sum(alpha_exp * outputs, 1) # (batch_size x hidden_size)

        # Véc tơ combinition tổng hợp
        c_t = torch.cat([c_local, c_global], 1) # batch_size x (2*hidden_size)
        c_t = self.ct_dropout(c_t)
        # Tính scores

        # Step 4: Decoder
        # embedding cho toàn bộ các item
        item_indices = torch.arange(self.n_items).to(device) # 1 x n_items
        item_embs = self.embedding(item_indices) # n_items x embedding_dim
        # reduce dimension by bi-linear projection
        B = self.b(item_embs).permute(1, 0) # (2*hidden_size) x n_items
        scores = torch.matmul(c_t, B) # batch_size x n_items
        # scores = self.sf(scores)
        return scores
    





def get_recall(indices, targets):
    """
    Tính toán chỉ số recall cho một tập hợp predictions và targets
    Args:
        indices (Bxk): torch.LongTensor. top-k indices được dự báo từ mô hình model.
        targets (B): torch.LongTensor. actual target indices.
    Returns:
        recall (float): the recall score
    """
    # copy targets k lần để trở thành kích thước Bxk
    targets = targets.view(-1, 1).expand_as(indices)
    # so sánh targets với indices để tìm ra vị trí mà khách hàng sẽ hit.
    hits = (targets == indices).to(device)
    hits = hits.double()
    if targets.size(0) == 0:
        return 0
    # Đếm số hit
    n_hits = torch.sum(hits)
    recall = n_hits / targets.size(0)
    return recall


def get_mrr(indices, targets):
    """
    Tính toán chỉ số MRR cho một tập hợp predictions và targets
    Args:
        indices (Bxk): torch.LongTensor. top-k indices được dự báo từ mô hình model.
        targets (B): torch.LongTensor. actual target indices.
    Returns:
        recall (float): the MRR score
    """
    tmp = targets.view(-1, 1)
    targets = tmp.expand_as(indices)
    hits = (targets == indices).to(device)
    hits = hits.double()
    if hits.sum() == 0:
      return 0
    argsort = []
    for i in np.arange(hits.shape[0]):
      index_col = torch.where(hits[i, :] == 1)[0]+1
      if index_col.shape[0] != 0:
        argsort.append(index_col.double())
    inv_argsort = [1/item for item in argsort]
    mrr = sum(inv_argsort)/hits.shape[0]
    return mrr


def evaluate(logits, targets, k=20):
    """
    Đánh giá model sử dụng Recall@K, MRR@K scores.
    Args:
        logits (B,C): torch.LongTensor. giá trị predicted logit cho itemId tiếp theo.
        targets (B): torch.LongTensor. actual target indices.
    Returns:
        recall (float): the recall score
        mrr (float): the mrr score
    """
    # Tìm ra indices của topk lớn nhất các giá trị dự báo.
    _, indices = torch.topk(logits, k, -1)
    recall = get_recall(indices, targets)
    mrr = get_mrr(indices, targets)
    if mrr == 0:
      mrr = torch.tensor(mrr, dtype = torch.float64).reshape(1)
    if recall == 0:
      recall = torch.tensor(recall, dtype = torch.float64)
    return recall, mrr\
    

def validate(valid_loader, model):
    model.eval()
    recalls = []
    mrrs = []

    with torch.no_grad():
        for seq, target, lens in valid_loader:
            seq = seq.to(device)
            target = target.to(device)
            outputs = model(seq, lens)
            logits = F.softmax(outputs, dim = 1)
            recall, mrr = evaluate(logits, target, k = 20)
            recall = recall.to(device)
             #print(recall)
            recalls.append(recall)
            # print(mrr)
            mrr = mrr.to(device)
            mrrs.append(mrr)
    # print(recalls, mrrs)
    # print(len(recalls), len(mrrs))

    # mean_recall = torch.mean(torch.stack(recalls))
    # print(recalls[:7])
    mean_recall = torch.mean(torch.stack(recalls))
    # print(mrrs[:7])
    mean_mrr = torch.mean(torch.stack(mrrs))

    return float(mean_recall), float(mean_mrr)

def trainForEpoch(train_loader, model, optimizer, epoch, num_epochs, criterion, log_aggr=1000):
    model.train()

    sum_epoch_loss = 0

    start = time.time()
    for i, (seq, target, lens) in enumerate(train_loader):
        seq = seq.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        outputs = model(seq, lens)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        sum_epoch_loss += loss_val

        iter_num = epoch * len(train_loader) + i + 1

        if i % log_aggr == 0:
            print('[TRAIN] epoch %d/%d  observation %d/%d batch loss: %.4f (avg %.4f) (%.2f im/s)'
                % (epoch , num_epochs, i, len(train_loader), loss_val, sum_epoch_loss / (i + 1),
                  len(seq) / (time.time() - start)))

        start = time.time()



args = {
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
    'valid_portion':0.1,
    'saved_data': 'NARM'
}



def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Loading data...')
    train_data, valid_data, test_data = load_data(root=args['dataset_path'])
    train_data = RecSysDataset(train_data)
    valid_data = RecSysDataset(valid_data)

    train_loader = DataLoader(train_data, batch_size = args['batch_size'], shuffle = True, collate_fn = collate_fn)
    valid_loader = DataLoader(valid_data, batch_size = args['batch_size'], shuffle = False, collate_fn = collate_fn)
    test_data = RecSysDataset(test_data)
    test_loader = DataLoader(test_data, batch_size = args['batch_size'], shuffle = False, collate_fn = collate_fn)
    print('Complete load data!')
    n_items = 22055
    model = NARM(hidden_size = args['hidden_size'], n_items = n_items, embedding_dim = args['embed_dim'], n_layers=2, dropout=0.25).to(device)
    print('complete load model!')

    if args['test'] == 'store_true':
        ckpt = torch.load('latest_checkpoint.pth.tar')
        model.load_state_dict(ckpt['state_dict'])
        recall, mrr = validate(test_loader, model)
        print("Test: Recall@{}: {:.4f}, MRR@{}: {:.4f}".format(args['topk'], recall, args['topk'], mrr))
        return model

    optimizer = optim.Adam(model.parameters(), args['lr'])
    criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size = args['lr_dc_step'], gamma = args['lr_dc'])
    best_result = [0,0]
    best_epoch = [0,0]
    print('start training!')
    for epoch in tqdm(range(args['epoch'])):
        # train for one epoch
        trainForEpoch(train_loader, model, optimizer, epoch, args['epoch'], criterion, log_aggr = 1000)
        scheduler.step(epoch = epoch)

        recall, mrr = validate(valid_loader, model)
        print('Epoch {} validation: Recall@{}: {:.4f}, MRR@{}: {:.4f} \n'.format(epoch, args['topk'], recall, args['topk'], mrr))
        if recall >= best_result[0]:
            best_result[0] = recall
            best_epoch[0] = epoch

            torch.save(model, args['saved_data'] + '/best_recall.pt')
        if mrr >= best_result[1]:
            best_result[1] = mrr
            best_epoch[1] = epoch

            torch.save(model, args['saved_data'] + '/best_mrr.pt')
        # store best loss and save a model checkpoint
        ckpt_dict = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_epoch': best_epoch,
            'best_result': best_result
        }
        # Save model checkpoint into 'latest_checkpoint.pth.tar'
        torch.save(ckpt_dict, args['dataset_path'] +  f'checkpoint.pth.tar')
   


if __name__ == '__main__':
    main()
    print('Training is done!')