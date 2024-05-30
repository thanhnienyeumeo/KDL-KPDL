from utils import *
import pickle
opt = {
    'dataset': 'yoochoose1_64',
    'batchSize': 75,
    'hiddenSize': 120,
    'epoch': 15,
    'lr': 0.001,
    'lr_dc': 0.1,
    'lr_dc_step': 3,
    'l2': 1e-5,
    'step': 1,
    'patience': 10,
    'nonhybrid': False,
    'validation': True,
    'valid_portion': 0.1,
    'dynamic': False,
    'saved_data': 'GCSAN',
    'mini': False,
    'model': False

}

test_data = pickle.load(open('datasets' + '/test.pkl', 'rb'))
test_data = Data(test_data, shuffle=False, opt=opt)

print(test_data.len_max)