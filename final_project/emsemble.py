
import pickle
from util import *
import numpy
filenames = [
    'base',
    'bt1b',
    'bt1k',
    'kaiming',
    'lstm1',
    'lstm2',
    'lstm3',
    'mxnet',
    'normal',
    'orthogonal',
    'tensorflow'
]

valid_data_dict = read_dir('./eval')
valid_data, valid_label = format_data(valid_data_dict)
print(valid_label.shape)
dataALL = np.zeros([0, 527])
for tmpname in filenames:
    data = pickle.load(open('result/' + tmpname + '/output.pkl', 'rb'))
    print(data.shape)
    if dataALL.shape[0] == 0:
        dataALL = data
    else:
        dataALL += data

    print(eval(data/len(filenames), valid_label))

with open('output.pkl', 'wb') as f:
    pickle.dump(data/len(filenames), f)