from __future__ import print_function
import numpy as np
import os

from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
import pickle
import gzip
from os import listdir, path
import sys

def read_dir(filedir):
    data_dict = {}
    if filedir == './big_train':
        for subfile in listdir(filedir):
            curfile = np.load(path.join(filedir, subfile), encoding='latin1')
            data_dict.update(curfile[()])
        return data_dict

    for subfile in listdir(filedir):
        with gzip.open(path.join(filedir,subfile), 'rb') as readsub:
            curfile = pickle.load(readsub, encoding='latin1')
            data_dict ={**data_dict, **curfile}
            # data_dict = dict(list(data_dict.items())+list(curfile.items()))
    return data_dict


def format_data(data_dict):

    n_data = len(data_dict)
    keys = data_dict.keys()
    data_arr = [data_dict[key][0] for key in keys]
    labels = [data_dict[key][1] for key in keys]
    data_arr = np.array(data_arr)
    label_arr = np.zeros((n_data, 527), dtype=np.float)
    for i in range(len(labels)):
        for idx in labels[i]:
            label_arr[i][idx] = 1
    return data_arr, label_arr


def eval(pred, label):
    mAP = average_precision_score(label, pred)
    mAUC = roc_auc_score(label, pred)
    return mAP, mAUC
# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
