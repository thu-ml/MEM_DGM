import numpy as np
import pickle as pkl
import cPickle as cPkl

import gzip
import tarfile
import fnmatch
import os
import urllib
from scipy.io import loadmat

def _unpickle(f):
    import cPickle
    fo = open(f, 'rb')
    d = cPickle.load(fo)
    fo.close()
    return d

def _get_datafolder_path():
    full_path = os.path.abspath('.')
    path = full_path +'/data'
    return path

def norb_random(datasets_dir=_get_datafolder_path()+"/norb_random/"):
    # random splited norb
    fname = 'norb_random.mat'
    x_train = loadmat(datasets_dir+fname)['x_train'].astype(np.float32)
    x_test = loadmat(datasets_dir+fname)['x_test'].astype(np.float32)
    return x_train, x_test

def CalTech101Silhouettes(datasets_dir=_get_datafolder_path()+"/caltech101_silhouettes/"):
    # get CalTech101Silhouettes dataset
    fname="caltech101_silhouettes_28_split1.mat"
    x_train = loadmat(datasets_dir+fname)['train_data'].astype(np.float32)
    x_valid = loadmat(datasets_dir+fname)['val_data'].astype(np.float32)
    x_test = loadmat(datasets_dir+fname)['test_data'].astype(np.float32)
    return x_train, x_valid, x_test

def svhn(datasets_dir=_get_datafolder_path()+"/svhn/", normalized=True, centered=False):
    zz = loadmat(datasets_dir+"train_32x32.mat")
    x_train = zz['X'].astype(np.float32)
    y_train = zz['y']
    x_train = x_train.transpose((2, 0, 1, 3)).reshape((3072, -1)).T
    zz = loadmat(datasets_dir+"test_32x32.mat")
    x_test = zz['X'].astype(np.float32)
    y_test = zz['y']
    x_test = x_test.transpose((2, 0, 1, 3)).reshape((3072, -1)).T
    if normalized:
        x_train = x_train / 256.0
        x_test = x_test / 256.0
    if centered:
        ave = x_train.sum(axis=0, keepdims=True)
        x_train = x_train - ave
        x_test = x_test - ave
    return x_train, y_train, x_test, y_test

def lfw(datasets_dir=_get_datafolder_path()+"/lfw/", normalize=True, colorImg=True, size='large'):
    fname = 'lfw_'
    if size == 'large':
        fname += '62x47'
        n_f = 62*47
    else:
        fname += '31x23'
        n_f = 31*23
    if not colorImg:
        fname += '_gray'
    fname += '.npy'
    if colorImg:
        x_train = np.load(datasets_dir+fname)
        print '---', x_train.shape
        x_train = x_train.swapaxes(1,3).swapaxes(2,3).reshape((-1, n_f*3))
    else:
        x_train = np.load(datasets_dir+fname).reshape((-1, n_f*1))
    if normalize:
        x_train = x_train/256.0
    return x_train

def ocr_letter(datasets_dir=_get_datafolder_path()+"/ocr_letter/"):
    # get ocr_letter dataset
    import h5py
    fname="ocr_letters.h5"
    f = h5py.File(datasets_dir+fname,'r')
    x_train = np.asarray(f['train']).astype(np.float32)
    x_valid = np.asarray(f['valid']).astype(np.float32)
    x_test = np.asarray(f['test']).astype(np.float32)
    return x_train, x_valid, x_test

def oivetti(datasets_dir=_get_datafolder_path()+"/oivetti/", normalize=True):
    '''
    url: http://www.cs.nyu.edu/~roweis/data.html
    Olivetti Faces [data/olivettifaces.mat] [picture]
    Grayscale faces 8 bit [0-255], a few images of several different people.
    400 total images, 64x64 size.
    From the Oivetti database at ATT.
    '''
    fname="olivettifaces.mat"
    x_train = loadmat(datasets_dir+fname)['faces'].astype(np.float32)
    x_train = x_train.T
    x_train = x_train.reshape((400,64,64))
    x_train = np.transpose(x_train, (0,2,1)).reshape((400,-1))
    if normalize:
        x_train = x_train/256.0
    return x_train

def cifar10(datasets_dir=_get_datafolder_path()+'/cifar10', num_val=5000, normalized=True, centered=True):
    # this code is largely cp from Kyle Kastner:
    #
    # https://gist.github.com/kastnerkyle/f3f67424adda343fef40

    url = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    data_file = os.path.join(datasets_dir, 'cifar-10-python.tar.gz')
    data_dir = os.path.join(datasets_dir, 'cifar-10-batches-py')

    if not os.path.exists(datasets_dir):
        os.makedirs(datasets_dir)

    if not os.path.isfile(data_file):
        urllib.urlretrieve(url, data_file)
        org_dir = os.getcwd()
        with tarfile.open(data_file) as tar:
            os.chdir(datasets_dir)
            tar.extractall()
        os.chdir(org_dir)

    train_files = []
    for filepath in fnmatch.filter(os.listdir(data_dir), 'data*'):
        train_files.append(os.path.join(data_dir, filepath))
    train_files = sorted(train_files, key=lambda x: x.split("_")[-1])

    test_file = os.path.join(data_dir, 'test_batch')

    x_train, targets_train = [], []
    for f in train_files:
        d = _unpickle(f)
        x_train.append(d['data'])
        targets_train.append(d['labels'])
    x_train = np.array(x_train, dtype='uint8')
    shp = x_train.shape
    x_train = x_train.reshape(shp[0] * shp[1], 3, 32, 32)
    targets_train = np.array(targets_train)
    targets_train = targets_train.ravel()

    d = _unpickle(test_file)
    x_test = d['data']
    targets_test = d['labels']
    x_test = np.array(x_test, dtype='uint8')
    x_test = x_test.reshape(-1, 3, 32, 32)
    targets_test = np.array(targets_test)
    targets_test = targets_test.ravel()
    
    if normalized:
    	x_train = x_train/256.0
    	x_test = x_test/256.0
    if centered:
    	avg = x_train.mean(axis=0,keepdims=True)
    	x_train = x_train - avg
    	x_test = x_test - avg

    if num_val is not None:
        perm = np.random.permutation(x_train.shape[0])
        x = x_train[perm]
        y = targets_train[perm]

        x_valid = x[:num_val]
        targets_valid = y[:num_val]
        x_train = x[num_val:]
        targets_train = y[num_val:]
        return (x_train, targets_train,
                x_valid, targets_valid,
                x_test, targets_test)
    else:
        return x_train, targets_train, x_test, targets_test

def omniglot_original(datasets_dir=_get_datafolder_path()+"/omniglot_original/"):
    # get omniglot dataset
    def combine(images):
        i_re = []
        for i in xrange(images.shape[0]):
            j_m = images[i][0].shape[0]
            for j in xrange(j_m):
                k_m = images[i][0][j][0].shape[0]
                for k in xrange(k_m):
                    i_re.append(images[i][0][j][0][k][0])
        return np.asarray(i_re)
    fname_train="data_background.mat"
    x_train = loadmat(datasets_dir+fname_train)['images']
    x_train = combine(x_train)
    fname_test="data_evaluation.mat"
    x_test = loadmat(datasets_dir+fname_test)['images']
    x_test = combine(x_test)
    return x_train, x_test

def omniglot(datasets_dir=_get_datafolder_path()+"/omniglot/"):
    # get omniglot dataset
    def reshape_data(data):
        return data.reshape((-1, 28, 28)).reshape((-1, 28*28), order='fortran')
    omni_raw = loadmat(datasets_dir+'chardata.mat')

    train_data = reshape_data(omni_raw['data'].T.astype('float32'))
    test_data = reshape_data(omni_raw['testdata'].T.astype('float32'))
    return train_data, test_data