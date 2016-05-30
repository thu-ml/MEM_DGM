import theano
theano.config.floatX = 'float32'
import matplotlib
matplotlib.use('Agg')
import theano.tensor as T
import numpy as np
import lasagne
from parmesan.distributions import log_stdnormal, log_normal2, log_bernoulli
from parmesan.layers import SampleLayer, NormalizeLayer, ScaleAndShiftLayer
from parmesan.datasets import load_mnist_realval, load_mnist_binarized
import matplotlib.pyplot as plt
import shutil, gzip, os, cPickle, time, math, operator, argparse

from layers.analysis_memory import (SeparateMemoryLayer, AttentionLayer, NormalizedAttentionLayer, SimpleCompositionLayer, LadderCompositionLayer)
from datasets import CalTech101Silhouettes, cifar10, ocr_letter
#from datasets_norb import load_numpy_subclasses

import  util.paramgraphics as paramgraphics
import scipy.io as sio

filename_script = os.path.basename(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("-dataset", type=str, 
        help="datasets sample|fixed|caltech", default="sample")
parser.add_argument("-eq_samples", type=int,
        help="number of samples for the expectation over q(z|x)", default=1)
parser.add_argument("-iw_samples", type=int,
        help="number of importance weighted samples", default=1)
parser.add_argument("-lr", type=float,
        help="learning rate", default=0.001)
parser.add_argument("-anneal_lr_factor", type=float,
        help="learning rate annealing factor", default=0.998)
parser.add_argument("-anneal_lr_epoch", type=float,
        help="larning rate annealing start epoch", default=1000)
parser.add_argument("-batch_norm", type=str,
        help="batch normalization", default='true')
parser.add_argument("-outfolder", type=str,
        help="output folder", default=os.path.join("results", os.path.splitext(filename_script)[0]))
parser.add_argument("-nonlin_enc", type=str,
        help="encoder non-linearity", default="rectify")
parser.add_argument("-nonlin_dec", type=str,
        help="decoder non-linearity", default="rectify")
parser.add_argument("-nlatent", type=int,
        help="number of stochastic latent units", default=100)
parser.add_argument("-batch_size", type=int,
        help="batch size", default=100)
parser.add_argument("-nepochs", type=int,
        help="number of epochs to train", default=3000)
parser.add_argument("-eval_epoch", type=int,
        help="epochs between evaluation of test performance", default=10)

# architecture and parameter
parser.add_argument("-com_type", type=str, default='ladder')
parser.add_argument("-atten_type", type=str, default='normalized')
parser.add_argument("-lre_type", type=str, default='norm')
parser.add_argument("-n_layers", type=int, default=2)
parser.add_argument("-n_hiddens", type=str, default='500,500')
parser.add_argument("-drops_enc", type=str, default='0,0')
parser.add_argument("-has_memory", type=str, default='0,1,1')
parser.add_argument("-has_lre", type=str, default='0,0,0')
parser.add_argument("-lambdas", type=str, default='0,0,0')
parser.add_argument("-n_slots", type=str, default='50,50,50')
parser.add_argument("-model_file", type=str, default=None)
parser.add_argument("-analysis_mode", type=str, default='imputation')
parser.add_argument("-imputation_mode", type=str, default='random')
parser.add_argument("-imputation_para", type=float, default=0.8)
parser.add_argument("-mem_activations_dir", type=str, default='data_analysis/mem_activations.mat')

args = parser.parse_args()

model_file = args.model_file
analysis_mode = args.analysis_mode
imputation_mode = args.imputation_mode
imputation_para = args.imputation_para
mem_activations_dir = args.mem_activations_dir
assert model_file is not None

has_memory = map(int, args.has_memory.split(','))
has_lre = map(int, args.has_lre.split(','))
n_hiddens = map(int, args.n_hiddens.split(','))
lambdas = map(float, args.lambdas.split(','))
n_slots = map(int, args.n_slots.split(','))
drops_enc = map(float, args.drops_enc.split(','))
n_layers = args.n_layers

assert len(n_hiddens) == n_layers
assert len(drops_enc) == n_layers
assert len(n_slots) == (n_layers+1)
assert len(has_lre) == (n_layers+1)
assert len(has_memory) == (n_layers+1)
assert len(lambdas) == (n_layers+1)


def get_nonlin(nonlin):
    if nonlin == 'rectify':
        return lasagne.nonlinearities.rectify
    elif nonlin == 'very_leaky_rectify':
        return lasagne.nonlinearities.very_leaky_rectify
    elif nonlin == 'tanh':
        return lasagne.nonlinearities.tanh
    else:
        raise ValueError('invalid non-linearity \'' + nonlin + '\'')

iw_samples = args.iw_samples   #number of importance weighted samples
eq_samples = args.eq_samples   #number of samples for the expectation over E_q(z|x)
lr = args.lr
anneal_lr_factor = args.anneal_lr_factor
anneal_lr_epoch = args.anneal_lr_epoch
batch_norm = args.batch_norm == 'true' or args.batch_norm == 'True'
nonlin_enc = get_nonlin(args.nonlin_enc)
nonlin_dec = get_nonlin(args.nonlin_dec)
latent_size = args.nlatent
dataset = args.dataset
batch_size = args.batch_size
num_epochs = args.nepochs
eval_epoch = args.eval_epoch

# result folder
res_out = args.outfolder
res_out += '_'
if sum(has_memory) == 0:
    res_out += 'baseline'
else:
    res_out+= 'ours'
res_out += '_'
res_out += dataset
res_out += '_'
res_out += analysis_mode
if analysis_mode == 'imputation':
    res_out += '_'
    res_out += str(imputation_mode)
    res_out += '_'
    res_out += str(imputation_para)
res_out += '_'
res_out += str(int(time.time()))

assert dataset in ['sample','fixed', 'caltech', 'norb_48', 'norb_96', 'cifar10', 'ocr_letter'], "dataset must be sample|fixed|caltech"

np.random.seed(1234)

### SET UP LOGFILE AND OUTPUT FOLDER
if not os.path.exists(res_out):
    os.makedirs(res_out)

# write commandline parameters to header of logfile
args_dict = vars(args)
sorted_args = sorted(args_dict.items(), key=operator.itemgetter(0))
description = []
description.append('######################################################')
description.append('# --Commandline Params--')
for name, val in sorted_args:
    description.append("# " + name + ":\t" + str(val))
description.append('######################################################')

shutil.copy(os.path.realpath(__file__), os.path.join(res_out, filename_script))
logfile = os.path.join(res_out, 'logfile.log')
model_out = os.path.join(res_out, 'model')
with open(logfile,'w') as f:
    for l in description:
        f.write(l + '\n')


sym_iw_samples = T.iscalar('iw_samples')
sym_eq_samples = T.iscalar('eq_samples')
sym_lr = T.scalar('lr')
sym_x = T.matrix('x')

if dataset in ['sample', 'fixed', 'caltech']:
    colorImg = False
    dim_input = (28,28)
    in_channels = 1
    generation_scale = False
    num_generation = 64
elif dataset == 'ocr_letter':
    colorImg = False
    dim_input = (16,8)
    in_channels = 1
    generation_scale = False
    num_generation = 64
elif dataset == 'norb_48':
    colorImg = False
    dim_input = (48,48)
    in_channels = 1
    generation_scale = True
    num_generation = 64
elif dataset == 'norb_96':
    colorImg = False
    dim_input = (96,96)
    in_channels = 1
    generation_scale = True
    num_generation = 25
elif dataset == 'cifar10':
    colorImg = True
    dim_input = (32,32)
    in_channels = 3
    generation_scale = True
    num_generation = 64
num_features = in_channels*dim_input[0]*dim_input[1]


def bernoullisample(x):
    return np.random.binomial(1,x,size=x.shape).astype(theano.config.floatX)

### LOAD DATA AND SET UP SHARED VARIABLES
if dataset == 'sample':
    print "Using real valued MNIST dataset to binomial sample dataset after every epoch "
    train_x, train_t, valid_x, valid_t, test_x, test_t = load_mnist_realval()
    #del train_t, valid_t, test_t
    preprocesses_dataset = bernoullisample
elif dataset == 'fixed':
    print "Using fixed binarized MNIST data"
    train_x, valid_x, test_x = load_mnist_binarized()
    preprocesses_dataset = lambda dataset: dataset #just a dummy function
elif dataset == 'caltech':
    print "Using CalTech101Silhouettes dataset"
    train_x, valid_x, test_x = CalTech101Silhouettes()
    preprocesses_dataset = lambda dataset: dataset #just a dummy function
elif dataset == 'norb_48':
    print "Using NORB dataset, size = 48"
    x, y = load_numpy_subclasses(size=48, normalize=True)
    x = x.T
    train_x = x[:24300]
    test_x = x[24300*2:24300*3] # only for debug, compare generation only
    del y
    preprocesses_dataset = lambda dataset: dataset #just a dummy function
elif dataset == 'norb_96':
    print "Using NORB dataset, size = 96"
    x, y = load_numpy_subclasses(size=96, normalize=True)
    x = x.T
    train_x = x[:24300]
    test_x = x[24300*2:24300*3] # only for debug, compare generation only
    del y
    preprocesses_dataset = lambda dataset: dataset #just a dummy function
elif dataset is 'ocr_letter':
    print "Using ocr_letter dataset"
    train_x, valid_x, test_x = ocr_letter()
    preprocesses_dataset = lambda dataset: dataset #just a dummy function

elif dataset == 'cifar10':
    print "Using CIFAR10 dataset"
    train_x, train_t, test_x, test_t = cifar10(num_val=None, normalized=True, centered=False)
    preprocesses_dataset = lambda dataset: dataset #just a dummy function
    train_x = train_x.reshape((-1,num_features))
    test_x = test_x.reshape((-1,num_features))
else:
    print 'Wrong dataset', dataset
    exit()

if dataset in ['sample', 'fixed', 'caltech', 'ocr_letter']:
    train_x = np.concatenate([train_x,valid_x])
if dataset == 'sample':
    train_t = np.concatenate([train_t,valid_t])

train_x = train_x.astype(theano.config.floatX)
test_x = test_x.astype(theano.config.floatX)

# do not preprocess data in testing
sh_x_train = theano.shared(train_x, borrow=True)
sh_x_test = theano.shared(test_x, borrow=True)


def batchnormlayer(l,num_units, nonlinearity, name, W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.)):
    l = lasagne.layers.DenseLayer(l, num_units=num_units, name="Dense-" + name, W=W, b=b, nonlinearity=None)
    l_n = NormalizeLayer(l,name="BN-" + name)
    l = ScaleAndShiftLayer(l_n,name="SaS-" + name)
    l = lasagne.layers.NonlinearityLayer(l,nonlinearity=nonlinearity,name="Nonlin-" + name)
    return l, l_n

def normaldenselayer(l,num_units, nonlinearity, name, W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.)):
    l = lasagne.layers.DenseLayer(l, num_units=num_units, name="Dense-" + name, W=W, b=b, nonlinearity=nonlinearity)
    return l, l

if batch_norm:
    print "Using batch Normalization - The current implementation calculates " \
          "the BN constants on the complete dataset in one batch. This might " \
          "cause memory problems on some GFX's"
    denselayer = batchnormlayer
else:
    denselayer = normaldenselayer

if args.com_type=='plus':
    compositelayer=SimpleCompositionLayer
elif args.com_type=='ladder':
    compositelayer=LadderCompositionLayer
else:
    raise ValueError('Unknown type of composition function.')

if args.atten_type=='normalized':
    attentionLayer=NormalizedAttentionLayer
elif args.atten_type=='unnormalized':
    attentionLayer=AttentionLayer
else:
    raise ValueError('Unknown type of attention function.')

def decoderlayer(l, has_memory, d_slots, n_slots, num_units, nonlinearity, name):
    if name == 'X_MU':
        h_g = lasagne.layers.DenseLayer(incoming=l, num_units=num_units, nonlinearity=nonlinearity, name=name)
    else:
        h_g, _ = denselayer(l=l, num_units=num_units, nonlinearity=nonlinearity, name=name)
    if has_memory == 1:
        # separated layers for analysis, slightly different with training
        # TODO 
        # make a unified version of train and analysis
        h_pro = attentionLayer(incoming=h_g, n_slots=n_slots, name='MEM_'+name)
        h_m = SeparateMemoryLayer(incoming=h_pro, n_slots=n_slots, d_slots=d_slots, nonlinearity_final=lasagne.nonlinearities.identity, name='MEM_'+name)
        if name == 'X_MU':
            h_g_next = compositelayer(h_g, h_m, nonlinearity_final=nonlinearity, name='COM_'+name)
        else:
            h_g_next = compositelayer(h_g, h_m, nonlinearity_final=nonlinearity, name='COM_'+name)
        return h_g_next, h_pro, h_g, h_m
    else:
        return h_g, None, None, None
        

### MODEL SETUP
# Recognition model q(z|x)
l_in = lasagne.layers.InputLayer((None, num_features))
l_enc = [l_in,]
f_enc = []
for i in xrange(n_layers):
    l, f = denselayer(l_enc[-1], num_units=n_hiddens[i], name='ENC_DENSE'+str(i+1), nonlinearity=nonlin_enc)
    if drops_enc[i] != 0:
        l = lasagne.layers.DropoutLayer(l, p=drops_enc[i])
    l_enc.append(l)
    f_enc.append(f)
l_mu = lasagne.layers.DenseLayer(l_enc[-1], num_units=latent_size, nonlinearity=lasagne.nonlinearities.identity, name='ENC_MU')
l_log_var = lasagne.layers.DenseLayer(l_enc[-1], num_units=latent_size, nonlinearity=lasagne.nonlinearities.identity, name='ENC_LOG_VAR')

#sample layer
l_z = SampleLayer(mu=l_mu, log_var=l_log_var, eq_samples=sym_eq_samples, iw_samples=sym_iw_samples)


# Generative model p(x|z)
l_dec = [l_z]
f_dec = []
p_dec = []
h_g_dec = []
h_m_dec = []
for i in reversed(xrange(n_layers)):
    l, l_pro, l_h_g, l_h_m = decoderlayer(l_dec[-1], has_memory[i+1], n_hiddens[i], n_slots[i+1], n_hiddens[i], nonlinearity=nonlin_dec, name='DEC_DENSE'+str(i+1))
    l_dec.append(l)
    f_dec.append(NormalizeLayer(l,name='BN-DEC_DENSE'+str(i+1)))
    if l_pro is not None:
        p_dec.append(l_pro)
        h_g_dec.append(l_h_g)
        h_m_dec.append(l_h_m)

if dataset in ['sample', 'fixed', 'caltech', 'ocr_letter']:
    l_dec_x_mu,_,_,_ = decoderlayer(l_dec[-1], has_memory[0], num_features, n_slots[0], num_features, nonlinearity=lasagne.nonlinearities.sigmoid, name='X_MU')
else:
    l_dec_x_mu,_,_,_ = decoderlayer(l_dec[-1], has_memory[0], num_features, n_slots[0], num_features, nonlinearity=lasagne.nonlinearities.identity, name='X_MU')
    # no memory for var
    l_dec_x_log_var,_,_,_ = decoderlayer(l_dec[-1], 0, num_features, n_slots[0], num_features, nonlinearity=lasagne.nonlinearities.identity, name='X_LOG_VAR')

if dataset in ['sample', 'fixed', 'caltech', 'ocr_letter']:
    # get output needed for evaluating of training i.e with noise if any
    z_train, z_mu_train, z_log_var_train, x_mu_train = lasagne.layers.get_output(
        [l_z, l_mu, l_log_var, l_dec_x_mu], sym_x, deterministic=False
    )

    # get output needed for evaluating of testing i.e without noise
    z_eval, z_mu_eval, z_log_var_eval, x_mu_eval = lasagne.layers.get_output(
        [l_z, l_mu, l_log_var, l_dec_x_mu], sym_x, deterministic=True
    )
else:
    # get output needed for evaluating of training i.e with noise if any
    z_train, z_mu_train, z_log_var_train, x_mu_train, x_log_var_train = lasagne.layers.get_output(
        [l_z, l_mu, l_log_var, l_dec_x_mu, l_dec_x_log_var], sym_x, deterministic=False
    )
    # get output needed for evaluating of testing i.e without noise
    z_eval, z_mu_eval, z_log_var_eval, x_mu_eval, x_log_var_eval = lasagne.layers.get_output(
        [l_z, l_mu, l_log_var, l_dec_x_mu, l_dec_x_log_var], sym_x, deterministic=True
    )

def latent_gaussian_x_gaussian(z, z_mu, z_log_var, x_mu, x_log_var, x, eq_samples, iw_samples, epsilon=1e-6):
    # reshape the variables so batch_size, eq_samples and iw_samples are separate dimensions
    z = z.reshape((-1, eq_samples, iw_samples, latent_size))
    x_mu = x_mu.reshape((-1, eq_samples, iw_samples, num_features))
    x_log_var = x_log_var.reshape((-1, eq_samples, iw_samples, num_features))

    # dimshuffle x, z_mu and z_log_var since we need to broadcast them when calculating the pdfs
    x = x.reshape((-1,num_features))
    x = x.dimshuffle(0, 'x', 'x', 1)                    # size: (batch_size, eq_samples, iw_samples, num_features)
    z_mu = z_mu.dimshuffle(0, 'x', 'x', 1)              # size: (batch_size, eq_samples, iw_samples, num_latent)
    z_log_var = z_log_var.dimshuffle(0, 'x', 'x', 1)    # size: (batch_size, eq_samples, iw_samples, num_latent)

    # calculate LL components, note that the log_xyz() functions return log prob. for indepenedent components separately 
    # so we sum over feature/latent dimensions for multivariate pdfs
    log_qz_given_x = log_normal2(z, z_mu, z_log_var).sum(axis=3)
    log_pz = log_stdnormal(z).sum(axis=3)
    #log_px_given_z = log_bernoulli(x, T.clip(x_mu, epsilon, 1 - epsilon)).sum(axis=3)
    log_px_given_z = log_normal2(x, x_mu, x_log_var).sum(axis=3)

    #all log_*** should have dimension (batch_size, eq_samples, iw_samples)
    # Calculate the LL using log-sum-exp to avoid underflow
    a = log_pz + log_px_given_z - log_qz_given_x    # size: (batch_size, eq_samples, iw_samples)
    a_max = T.max(a, axis=2, keepdims=True)         # size: (batch_size, eq_samples, 1)

    LL = T.mean(a_max) + T.mean( T.log( T.mean(T.exp(a-a_max), axis=2) ) )

    return LL, T.mean(log_qz_given_x), T.mean(log_pz), T.mean(log_px_given_z)


def latent_gaussian_x_bernoulli(z, z_mu, z_log_var, x_mu, x, eq_samples, iw_samples, epsilon=1e-6):
    """
    Latent z       : gaussian with standard normal prior
    decoder output : bernoulli

    When the output is bernoulli then the output from the decoder
    should be sigmoid. The sizes of the inputs are
    z: (batch_size*eq_samples*iw_samples, num_latent)
    z_mu: (batch_size, num_latent)
    z_log_var: (batch_size, num_latent)
    x_mu: (batch_size*eq_samples*iw_samples, num_features)
    x: (batch_size, num_features)

    Reference: Burda et al. 2015 "Importance Weighted Autoencoders"
    """

    # reshape the variables so batch_size, eq_samples and iw_samples are separate dimensions
    z = z.reshape((-1, eq_samples, iw_samples, latent_size))
    x_mu = x_mu.reshape((-1, eq_samples, iw_samples, num_features))

    # dimshuffle x, z_mu and z_log_var since we need to broadcast them when calculating the pdfs
    x = x.dimshuffle(0, 'x', 'x', 1)                    # size: (batch_size, eq_samples, iw_samples, num_features)
    z_mu = z_mu.dimshuffle(0, 'x', 'x', 1)              # size: (batch_size, eq_samples, iw_samples, num_latent)
    z_log_var = z_log_var.dimshuffle(0, 'x', 'x', 1)    # size: (batch_size, eq_samples, iw_samples, num_latent)

    # calculate LL components, note that the log_xyz() functions return log prob. for indepenedent components separately 
    # so we sum over feature/latent dimensions for multivariate pdfs
    log_qz_given_x = log_normal2(z, z_mu, z_log_var).sum(axis=3)
    log_pz = log_stdnormal(z).sum(axis=3)
    log_px_given_z = log_bernoulli(x, T.clip(x_mu, epsilon, 1 - epsilon)).sum(axis=3)

    #all log_*** should have dimension (batch_size, eq_samples, iw_samples)
    # Calculate the LL using log-sum-exp to avoid underflow
    a = log_pz + log_px_given_z - log_qz_given_x    # size: (batch_size, eq_samples, iw_samples)
    a_max = T.max(a, axis=2, keepdims=True)         # size: (batch_size, eq_samples, 1)

    # LL is calculated using Eq (8) in Burda et al.
    # Working from inside out of the calculation below:
    # T.exp(a-a_max): (batch_size, eq_samples, iw_samples)
    # -> subtract a_max to avoid overflow. a_max is specific for  each set of
    # importance samples and is broadcasted over the last dimension.
    #
    # T.log( T.mean(T.exp(a-a_max), axis=2) ): (batch_size, eq_samples)
    # -> This is the log of the sum over the importance weighted samples
    #
    # The outer T.mean() computes the mean over eq_samples and batch_size
    #
    # Lastly we add T.mean(a_max) to correct for the log-sum-exp trick
    LL = T.mean(a_max) + T.mean( T.log( T.mean(T.exp(a-a_max), axis=2) ) )

    return LL, T.mean(log_qz_given_x), T.mean(log_pz), T.mean(log_px_given_z)

# LOWER BOUNDS
if dataset in ['sample', 'fixed', 'caltech', 'ocr_letter']:
    LL_train, log_qz_given_x_train, log_pz_train, log_px_given_z_train = latent_gaussian_x_bernoulli(
        z_train, z_mu_train, z_log_var_train, x_mu_train, sym_x, eq_samples=sym_eq_samples, iw_samples=sym_iw_samples)

    LL_eval, log_qz_given_x_eval, log_pz_eval, log_px_given_z_eval = latent_gaussian_x_bernoulli(
        z_eval, z_mu_eval, z_log_var_eval, x_mu_eval, sym_x, eq_samples=sym_eq_samples, iw_samples=sym_iw_samples)
else:
    LL_train, log_qz_given_x_train, log_pz_train, log_px_given_z_train = latent_gaussian_x_gaussian(
        z_train, z_mu_train, z_log_var_train, x_mu_train, x_log_var_train, sym_x, eq_samples=sym_eq_samples, iw_samples=sym_iw_samples)
    LL_eval, log_qz_given_x_eval, log_pz_eval, log_px_given_z_eval = latent_gaussian_x_gaussian(
        z_eval, z_mu_eval, z_log_var_eval, x_mu_eval, x_log_var_eval, sym_x, eq_samples=sym_eq_samples, iw_samples=sym_iw_samples)

#some sanity checks that we can forward data through the model
X = np.ones((batch_size, num_features), dtype=theano.config.floatX) # dummy data for testing the implementation

print "OUTPUT SIZE OF l_z using BS=%d, latent_size=%d, sym_iw_samples=%d, sym_eq_samples=%d --"\
      %(batch_size, latent_size, iw_samples, eq_samples), \
    lasagne.layers.get_output(l_z,sym_x).eval(
    {sym_x: X, sym_iw_samples: np.int32(iw_samples),
     sym_eq_samples: np.int32(eq_samples)}).shape

#print "log_pz_train", log_pz_train.eval({sym_x:X, sym_iw_samples: np.int32(iw_samples),sym_eq_samples:np.int32(eq_samples)}).shape
#print "log_px_given_z_train", log_px_given_z_train.eval({sym_x:X, sym_iw_samples: np.int32(iw_samples), sym_eq_samples:np.int32(eq_samples)}).shape
#print "log_qz_given_x_train", log_qz_given_x_train.eval({sym_x:X, sym_iw_samples: np.int32(iw_samples), sym_eq_samples:np.int32(eq_samples)}).shape
#print "lower_bound_train", LL_train.eval({sym_x:X, sym_iw_samples: np.int32(iw_samples), sym_eq_samples:np.int32(eq_samples)}).shape


# get all parameters
if dataset in ['sample', 'fixed', 'caltech', 'ocr_letter']:
    params = lasagne.layers.get_all_params([l_dec_x_mu], trainable=True)
    for p in params:
        print p, p.get_value().shape
    params_count = lasagne.layers.count_params([l_dec_x_mu], trainable=True)
else:
    params = lasagne.layers.get_all_params([l_dec_x_mu, l_dec_x_log_var], trainable=True)
    for p in params:
        print p, p.get_value().shape
    params_count = lasagne.layers.count_params([l_dec_x_mu, l_dec_x_log_var], trainable=True)
print 'Number of parameters:', params_count

# random generation for visualization
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
srng_ran = RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))
srng_ran_share = theano.tensor.shared_randomstreams.RandomStreams(1234)
sym_nimages = T.iscalar('nimages')

ran_z = srng_ran.normal((sym_nimages,latent_size))
if dataset in ['sample', 'fixed', 'caltech', 'ocr_letter']:
    random_x_mean = lasagne.layers.get_output(l_dec_x_mu, {l_z:ran_z}, deterministic=True)
    random_x = srng_ran_share.binomial(n=1, p=random_x_mean, dtype=theano.config.floatX)
else:
    random_x_mean, random_x_log_var = lasagne.layers.get_output([l_dec_x_mu, l_dec_x_log_var], {l_z:ran_z}, deterministic=True)
    random_x = srng_ran_share.normal(size=(sym_nimages,num_features), avg=random_x_mean, std=T.exp(0.5*random_x_log_var))
generate_model = theano.function(inputs=[sym_nimages], outputs=[random_x_mean, random_x])


# local reconstruction error 
if args.lre_type == 'norm':
    activation_enc = lasagne.layers.get_output(
        f_enc, sym_x, deterministic=False
    )
    activation_dec = lasagne.layers.get_output(
        f_dec, sym_x, deterministic=False
    )
else:
    activation_enc = lasagne.layers.get_output(
        l_enc[1:], sym_x, deterministic=False
    )
    activation_dec = lasagne.layers.get_output(
        l_dec[1:], sym_x, deterministic=False
    )
# averaged dec activations for single sample
for i in xrange(n_layers):
    activation_dec[i] = activation_dec[i].reshape((batch_size, eq_samples*iw_samples, -1)).mean(axis=1)
cost = -LL_train
for i in xrange(n_layers):
    if has_lre[i+1] == 1:
        cost += lambdas[i+1]*T.sqr(activation_enc[i].flatten(2) - activation_dec[n_layers - i - 1].flatten(2)).mean(axis=1).mean()
if has_lre[0] == 1:
    cost += lambdas[0]*T.sqr(x_mu_train.flatten(2) - sym_x.flatten(2)).mean(axis=1).mean()

# note the minus because we want to push up the lowerbound
grads = T.grad(cost, params)
clip_grad = 1
max_norm = 5
mgrads = lasagne.updates.total_norm_constraint(grads,max_norm=max_norm)
cgrads = [T.clip(g, -clip_grad, clip_grad) for g in mgrads]

updates = lasagne.updates.adam(cgrads, params, beta1=0.9, beta2=0.999, epsilon=1e-4, learning_rate=sym_lr)

# Helper symbolic variables to index into the shared train and test data
sym_index = T.iscalar('index')
sym_batch_size = T.iscalar('batch_size')
batch_slice = slice(sym_index * sym_batch_size, (sym_index + 1) * sym_batch_size)

train_model = theano.function([sym_index, sym_batch_size, sym_lr, sym_eq_samples, sym_iw_samples], [LL_train, cost+LL_train, log_qz_given_x_train, log_pz_train, log_px_given_z_train, z_mu_train, z_log_var_train],
                              givens={sym_x: sh_x_train[batch_slice]},
                              updates=updates)

test_model = theano.function([sym_index, sym_batch_size, sym_eq_samples, sym_iw_samples], [LL_eval, log_qz_given_x_eval, log_pz_eval, log_px_given_z_eval],
                              givens={sym_x: sh_x_test[batch_slice]})


features_layer = l_enc[-1]
features = lasagne.layers.get_output(features_layer, sym_x, deterministic=True)

train_features = theano.function([sym_index, sym_batch_size], [features],
                              givens={sym_x: sh_x_train[batch_slice]})
test_features = theano.function([sym_index, sym_batch_size], [features],
                              givens={sym_x: sh_x_test[batch_slice]})

real_x = T.matrix('real_x')
p_label = T.matrix('p_label')
x_denoised = lasagne.layers.get_output(l_dec_x_mu, sym_x, deterministic=False)
x_denoised = p_label*real_x+(1-p_label)*x_denoised
mse = ((real_x - x_denoised)**2).sum()

denoise_model = theano.function([sym_index, sym_batch_size, sym_eq_samples, sym_iw_samples, sym_x, p_label], [x_denoised, mse],
                                givens={real_x: sh_x_test[batch_slice]})

if analysis_mode in ['statis_computation', 'visualization', 'visualization0', 'sparse_over_class']:
    p1, p2 = lasagne.layers.get_output(p_dec, sym_x, deterministic=True)

    train_probs = theano.function([sym_index, sym_batch_size, sym_eq_samples, sym_iw_samples], [p1,p2],
                                  givens={sym_x: sh_x_train[batch_slice]})
    test_probs = theano.function([sym_index, sym_batch_size, sym_eq_samples, sym_iw_samples], [p1,p2],
                                  givens={sym_x: sh_x_test[batch_slice]})

    input_z = T.matrix('input_z')
    h_m2 = T.matrix('h_m2')
    h_m1 = T.matrix('h_m1')

    mean_with_mem = lasagne.layers.get_output(l_dec_x_mu, {l_z:input_z}, deterministic=True)
    mean_without_mem1 = lasagne.layers.get_output(l_dec_x_mu, {l_z:input_z,h_m_dec[-1]:h_m1}, deterministic=True)
    mean_without_mem2 = lasagne.layers.get_output(l_dec_x_mu, {l_z:input_z,h_m_dec[-1]:h_m1,h_m_dec[-2]:h_m2}, deterministic=True)

    visualization_model = theano.function(inputs=[input_z, h_m2, h_m1], outputs=[mean_with_mem, mean_without_mem1, mean_without_mem2])

input_z = T.matrix('input_z')
zero_h_m = T.matrix('zero_h_m')
zero_h_g = T.matrix('zero_h_g')
iden_prob = T.matrix('iden_prob')
'''
if dataset in ['sample', 'fixed', 'caltech', 'ocr_letter']:
    mean_with_mem = lasagne.layers.get_output(l_dec_x_mu, {l_z:input_z}, deterministic=True)
    mean_without_mem = lasagne.layers.get_output(l_dec_x_mu, {l_z:input_z,h_m_dec[-1]:zero_h_m}, deterministic=True)
    mean_mem = lasagne.layers.get_output(l_dec_x_mu, {l_z:input_z,h_g_dec[-1]:zero_h_g}, deterministic=True) 
    probs = lasagne.layers.get_output(p_dec[-1], {l_z:input_z}, deterministic=True) 
    
else:
    mean_with_mem, _ = lasagne.layers.get_output([l_dec_x_mu, l_dec_x_log_var], {l_z:input_z}, deterministic=True)
    mean_without_mem, _ = lasagne.layers.get_output([l_dec_x_mu, l_dec_x_log_var], {l_z:input_z,h_m_dec[-1]:zero_h_m}, deterministic=True)
    mean_mem, _ = lasagne.layers.get_output([l_dec_x_mu, l_dec_x_log_var], {l_z:input_z,h_g_dec[-1]:zero_h_g}, deterministic=True)
    probs = lasagne.layers.get_output(p_dec[-1], {l_z:input_z}, deterministic=True) 
    
visualization_model_gene = theano.function(inputs=[input_z, zero_h_g, zero_h_m], outputs=[mean_with_mem, mean_without_mem, mean_mem, probs])

if dataset in ['sample', 'fixed', 'caltech', 'ocr_letter']:
    mean_with_mem = lasagne.layers.get_output(l_dec_x_mu,{l_in:sym_x} ,deterministic=True)
    mean_without_mem = lasagne.layers.get_output(l_dec_x_mu, {l_in:sym_x,h_m_dec[-1]:zero_h_m}, deterministic=True)
    mean_mem = lasagne.layers.get_output(l_dec_x_mu, {l_in:sym_x,h_g_dec[-1]:zero_h_g}, deterministic=True) 
    probs = lasagne.layers.get_output(p_dec[-1], {l_in:sym_x}, deterministic=True) 
    
else:
    mean_with_mem, _ = lasagne.layers.get_output([l_dec_x_mu, l_dec_x_log_var], {l_in:sym_x}, deterministic=True)
    mean_without_mem, _ = lasagne.layers.get_output([l_dec_x_mu, l_dec_x_log_var], {l_in:sym_x,h_m_dec[-1]:zero_h_m}, deterministic=True)
    mean_mem, _ = lasagne.layers.get_output([l_dec_x_mu, l_dec_x_log_var], {l_in:sym_x,h_g_dec[-1]:zero_h_g}, deterministic=True)
    probs = lasagne.layers.get_output(p_dec[-1], {l_in:sym_x}, deterministic=True) 
    
visualization_model_reco= theano.function(inputs=[sym_eq_samples, sym_iw_samples, sym_x, zero_h_g, zero_h_m], outputs=[mean_with_mem, mean_without_mem, mean_mem, probs])
'''
if dataset in ['sample', 'fixed', 'caltech', 'ocr_letter']:
    mean_mem = lasagne.layers.get_output(l_dec_x_mu, {p_dec[-1]:iden_prob,h_g_dec[-1]:zero_h_g}, deterministic=True) 
    mem = lasagne.layers.get_output(h_m_dec[-1], {p_dec[-1]:iden_prob,h_g_dec[-1]:zero_h_g}, deterministic=True) 
    
else:
    mean_mem, _ = lasagne.layers.get_output([l_dec_x_mu, l_dec_x_log_var], {p_dec[-1]:iden_prob,h_g_dec[-1]:zero_h_g}, deterministic=True)
    mem = lasagne.layers.get_output(p_dec[-1], {p_dec[-1]:iden_prob,h_g_dec[-1]:zero_h_g}, deterministic=True) 
    
visualization_model_mem = theano.function(inputs=[zero_h_g, iden_prob], outputs=[mean_mem, mem])

if batch_norm:
    collect_out = lasagne.layers.get_output(l_dec_x_mu, sym_x, deterministic=True, collect=True)
    f_collect = theano.function([sym_eq_samples, sym_iw_samples],
                                [collect_out],
                                givens={sym_x: sh_x_train})

def test_epoch(eq_samples, iw_samples, batch_size):
    n_test_batches = test_x.shape[0] / batch_size
    costs, log_qz_given_x,log_pz,log_px_given_z = [],[],[],[]
    for i in range(n_test_batches):
        cost_batch, log_qz_given_x_batch, log_pz_batch, log_px_given_z_batch = test_model(i, batch_size, eq_samples, iw_samples)
        costs += [cost_batch]
        log_qz_given_x += [log_qz_given_x_batch]
        log_pz += [log_pz_batch]
        log_px_given_z += [log_px_given_z_batch]
    return np.mean(costs), np.mean(log_qz_given_x), np.mean(log_pz), np.mean(log_px_given_z)

def get_test_f(batch_size):
    n_test_batches = test_x.shape[0] / batch_size
    features = []
    for i in range(n_test_batches):
        features_batch = test_features(i, batch_size)
        features += [features_batch]
    return np.concatenate(features).reshape((-1,500))

def get_train_f(batch_size):
    n_train_batches = train_x.shape[0] / batch_size
    features = []
    for i in range(n_train_batches):
        features_batch = train_features(i, batch_size)
        features += [features_batch]
    return np.concatenate(features).reshape((-1,500))

def get_test_p(batch_size):
    n_test_batches = test_x.shape[0] / batch_size
    p1s,p2s = [], []
    for i in range(n_test_batches):
        p1s_batch, p2s_batch = test_probs(i, batch_size,1,1)
        p1s += [p1s_batch]
        p2s += [p2s_batch]
    return np.concatenate(p1s).reshape((-1,n_slots[-1])), np.concatenate(p2s).reshape((-1,n_slots[-2]))

def get_train_p(batch_size):
    n_train_batches = train_x.shape[0] / batch_size
    p1s,p2s = [], []
    for i in range(n_train_batches):
        p1s_batch, p2s_batch = train_probs(i, batch_size,1,1)
        p1s += [p1s_batch]
        p2s += [p2s_batch]
    return np.concatenate(p1s).reshape((-1,n_slots[-1])), np.concatenate(p2s).reshape((-1,n_slots[-2]))

def test_denoise(x_p, p_l, batch_size):
    n_test_batches = test_x.shape[0] / batch_size
    x_d=[]
    mse=0
    for i in range(n_test_batches):
        x_d_batch, mse_batch = denoise_model(i, batch_size,1,1, x_p[i*batch_size:(i+1)*batch_size,:], p_l[i*batch_size:(i+1)*batch_size,:])
        x_d += [x_d_batch]
        mse += mse_batch

    return np.concatenate(x_d), mse


# load model
print 'Loading model'
f = gzip.open(model_file, 'rb')
model_params_load = cPickle.load(f)
model_params = []
model_names = []
for p in model_params_load:
    model_names.append(p.name)
    model_params.append(np.asarray(p.get_value()).astype(np.float32))

# exchange the order of params because analysis use separated layers
# TODO 
# make a unified version of train and analysis

#if 'MEM_DEC_DENSE1.M' in model_names:
#    a, b = model_names.index('MEM_DEC_DENSE1.M'), model_names.index('MEM_DEC_DENSE1.b')
#    model_params[b], model_params[a] = model_params[a], model_params[b]
#if 'MEM_DEC_DENSE2.M' in model_names:
#    a, b = model_names.index('MEM_DEC_DENSE2.M'), model_names.index('MEM_DEC_DENSE2.b')
#    model_params[b], model_params[a] = model_params[a], model_params[b]


# set all parameters
if dataset in ['sample', 'fixed', 'caltech', 'ocr_letter']:
    lasagne.layers.set_all_param_values([l_dec_x_mu], model_params)
else:
    lasagne.layers.set_all_param_values([l_dec_x_mu, l_dec_x_log_var], model_params)

'''
# output the log likelihood
LL_test1, log_qz_given_x_test1, log_pz_test1, log_px_given_z_test1 = [],[],[],[]
LL_test5000, log_qz_given_x_test5000, log_pz_test5000, log_px_given_z_test5000 = [],[],[],[]

if dataset not in ['norb_48', 'norb_96']:
    print "calculating LL eq=1, iw=5000"
    test_out5000 = test_epoch(1, 5000, batch_size=5) # smaller batch size to reduce memory requirements
    LL_test5000 += [test_out5000[0]]
    log_qz_given_x_test5000 += [test_out5000[1]]
    log_pz_test5000 += [test_out5000[2]]
    log_px_given_z_test5000 += [test_out5000[3]]
print "calculating LL eq=1, iw=1"
test_out1 = test_epoch(1, 1, batch_size=50)
LL_test1 += [test_out1[0]]
log_qz_given_x_test1 += [test_out1[1]]
log_pz_test1 += [test_out1[2]]
log_px_given_z_test1 += [test_out1[3]]

if dataset not in ['norb_48', 'norb_96']:
    line = "  EVAL-L1:\tCost=%.5f\tlogq(z|x)=%.5f\tlogp(z)=%.5f\tlogp(x|z)=%.5f\n" %(LL_test1[-1], log_qz_given_x_test1[-1], log_pz_test1[-1], log_px_given_z_test1[-1]) + \
           "  EVAL-L5000:\tCost=%.5f\tlogq(z|x)=%.5f\tlogp(z)=%.5f\tlogp(x|z)=%.5f" %(LL_test5000[-1], log_qz_given_x_test5000[-1], log_pz_test5000[-1], log_px_given_z_test5000[-1])
else:
    line = "  EVAL-L1:\tCost=%.5f\tlogq(z|x)=%.5f\tlogp(z)=%.5f\tlogp(x|z)=%.5f" %(LL_test1[-1], log_qz_given_x_test1[-1], log_pz_test1[-1], log_px_given_z_test1[-1])

print line
'''

# imputation
if analysis_mode == 'imputation':
    assert imputation_mode in ['random', 'half', 'rectangle']
    if dataset == 'sample':
        filename = 'data_imputation/'
    elif dataset == 'norb_96':
        filename = 'data_imputation/norb96_'
    if imputation_mode == 'random':
        filename+='type_4_params_'+str(int(imputation_para*100))+'_noise_rawdata.mat'
    elif imputation_mode == 'half':
        filename+='type_5_params_0_14_noise_rawdata.mat'
    elif imputation_mode == 'rectangle':
        filename+='type_3_params_'+str(int(imputation_para))+'_noise_rawdata.mat'
    else:
        pass
    print 'loading data'
    zz = sio.loadmat(filename)
    data_train = zz['z_train'].T
    data = zz['z_test_original'].T
    print data.shape
    data_perturbed = zz['z_test'].T
    print data_perturbed.shape
    pertub_label = zz['pertub_label'].astype(np.float32).T
    print pertub_label.shape
    pertub_number = float(np.sum(1-pertub_label))
    print pertub_number

    denoise_epochs = 100
    visualization_epochs = 20
    num_vis = 100
    num_vis1 = 64
    images = data[:num_vis,:]

    image = paramgraphics.mat_to_img(data[:num_vis1,:].T, dim_input, colorImg=colorImg, scale=True)
    image.save(os.path.join(res_out, 'data.png'), 'PNG')

    image = paramgraphics.mat_to_img(data_perturbed[:num_vis1,:].T, dim_input, colorImg=colorImg, scale=True)
    image.save(os.path.join(res_out, 'before_denoise.png'), 'PNG')

    for i in xrange(denoise_epochs):
        data_perturbed = data_perturbed.astype(np.float32)
        if i < visualization_epochs:
            images = np.vstack((images, data_perturbed[:num_vis,:]))
        data_perturbed, mse = test_denoise(data_perturbed, pertub_label, 1000)
        print mse / pertub_number
        with open(logfile,'a') as f:
            f.write(str(mse / pertub_number) + "\n")

    #tile_shape = (visualization_epochs+1, num_vis)
    tile_shape = (num_vis, visualization_epochs+1)
    images = images.reshape((-1,num_vis,num_features))
    images = np.transpose(images, (1,0,2)).reshape((-1, num_features))

    image = paramgraphics.mat_to_img(data_perturbed[:num_vis1,:].T, dim_input, colorImg=colorImg, scale=True)
    image.save(os.path.join(res_out, 'after_denoise.png'), 'PNG')

    sio.savemat(os.path.join(res_out, 'visualization_procedure.mat'), {'data': images})
    image = paramgraphics.mat_to_img(images.T, dim_input, colorImg=colorImg, tile_shape=tile_shape, scale=True)
    image.save(os.path.join(res_out, 'visualization_procedure.png'), 'PNG')
    

elif analysis_mode == 'visualization': 
    # visualization with memory, without memory1 and without memory2
    # setting information from memory to vectors filled with 1s
    print 'Visualizing...'
    num_vis = 20
    input_z = np.random.normal(loc=0, scale=1, size=(num_vis, latent_size)).astype(np.float32)
    h_m1 = np.ones((num_vis, n_hiddens[0])).astype(np.float32)
    h_m2 = np.ones((num_vis, n_hiddens[1])).astype(np.float32)
    i_with_mem, i_without_mem1, i_without_mem2 = visualization_model(input_z, h_m2, h_m1)
    
    x_mean = np.vstack((i_with_mem, i_without_mem1, i_without_mem2))
    tile_shape = (3, num_vis)
    image = paramgraphics.mat_to_img(x_mean.T, dim_input, colorImg=colorImg, tile_shape=tile_shape, scale=True)
    image.save(os.path.join(res_out, 'visualization.png'), 'PNG')
    image = paramgraphics.mat_to_img(x_mean.T, dim_input, colorImg=colorImg, tile_shape=tile_shape, scale=False)
    image.save(os.path.join(res_out, 'visualization_no_scale.png'), 'PNG')

elif analysis_mode == 'visualization0': 
    # visualization with memory, without memory1 and without memory2
    # setting information from memory to vectors filled with 0s
    print 'Visualizing...'
    num_vis = 20
    input_z = np.random.normal(loc=0, scale=1, size=(num_vis, latent_size)).astype(np.float32)
    h_m1 = np.zeros((num_vis, n_hiddens[0])).astype(np.float32)
    h_m2 = np.zeros((num_vis, n_hiddens[1])).astype(np.float32)
    i_with_mem, i_without_mem1, i_without_mem2 = visualization_model(input_z, h_m2, h_m1)
    
    x_mean = np.vstack((i_with_mem, i_without_mem1, i_without_mem2))
    tile_shape = (3, num_vis)
    image = paramgraphics.mat_to_img(x_mean.T, dim_input, colorImg=colorImg, tile_shape=tile_shape, scale=True)
    image.save(os.path.join(res_out, 'visualization.png'), 'PNG')
    image = paramgraphics.mat_to_img(x_mean.T, dim_input, colorImg=colorImg, tile_shape=tile_shape, scale=False)
    image.save(os.path.join(res_out, 'visualization_no_scale.png'), 'PNG')

elif analysis_mode == 'sparse_over_class':
    def compute_statis(probs, labels, n_c=10):
        # normalize to achieve actual probability
        normlizer = probs.sum(axis=1, keepdims=True)
        print normlizer.max(), normlizer.min()
        activations = np.zeros((n_c, probs.shape[1]))
        for i in xrange(n_c):
            label_i = np.asarray(np.where(labels == i)).flatten()
            activations[i,:] = (probs[label_i,:]).mean(axis=0)
        tmp = activations
        activations = np.repeat(activations, 5, axis=0)
        activations = np.repeat(activations, 5, axis=1)
        return tmp, activations

    p2_test, p1_test = get_test_p(1000)
    print p2_test.shape
    print p1_test.shape

    colorImg = False
    tile_shape = (1,1)
    scale = True

    print 'Test-MEM 2'
    tmp2, mem2_test = compute_statis(p2_test, test_t)
    image = paramgraphics.mat_to_img(mem2_test.reshape((10*5*30*5,-1)), dim_input=(10*5,30*5), colorImg=colorImg, tile_shape=tile_shape, scale=scale)
    image.save(os.path.join(res_out, 'test2.png'), 'PNG')
    print tmp2.shape

    print 'Test-MEM 1'
    tmp1, mem1_test = compute_statis(p1_test, test_t)
    image = paramgraphics.mat_to_img(mem1_test.reshape((10*5*70*5,-1)), dim_input=(10*5,70*5), colorImg=colorImg, tile_shape=tile_shape, scale=scale)
    image.save(os.path.join(res_out, 'test1.png'), 'PNG')
    print tmp1.shape
    
    sio.savemat(os.path.join(res_out,'mem_activations.mat'), {'mem1_test' : tmp1, 'mem2_test' : tmp2})

elif analysis_mode == 'statis_computation':
    assert has_memory[1] == 1
    def compute_statis(probs, labels, n_c=10):
        # normalize to achieve actual probability
        normlizer = probs.sum(axis=1, keepdims=True)
        probs = probs / normlizer
        activations = np.zeros((n_c, probs.shape[1]))
        for i in xrange(n_c):
            label_i = np.asarray(np.where(labels == i)).flatten()
            activations[i,:] = (probs[label_i,:]).mean(axis=0)
        #activations = np.repeat(activations, 4, axis=0)
        #activations = np.repeat(activations, 4, axis=1)
        return activations

    p2_train, p1_train = get_train_p(1000)
    p2_test, p1_test = get_test_p(1000)
    print p2_train.shape
    print p1_train.shape
    print p2_test.shape
    print p1_test.shape

    colorImg = False
    tile_shape = (1,1)
    scale = True

    print 'Train-MEM 2'
    mem2_train = compute_statis(p2_train, train_t)
    image = paramgraphics.mat_to_img(mem2_train.reshape((10*30,-1)), dim_input=(10,30), colorImg=colorImg, tile_shape=tile_shape, scale=scale)
    image.save(os.path.join(res_out, 'train2.png'), 'PNG')
    print 'Train-MEM 1'
    mem1_train = compute_statis(p1_train, train_t)
    image = paramgraphics.mat_to_img(mem1_train.reshape((10*70,-1)), dim_input=(10,70), colorImg=colorImg, tile_shape=tile_shape, scale=scale)
    image.save(os.path.join(res_out, 'train1.png'), 'PNG')
    print 'Test-MEM 2'
    mem2_test = compute_statis(p2_test, test_t)
    image = paramgraphics.mat_to_img(mem2_test.reshape((10*30,-1)), dim_input=(10,30), colorImg=colorImg, tile_shape=tile_shape, scale=scale)
    image.save(os.path.join(res_out, 'test2.png'), 'PNG')
    print 'Test-MEM 1'
    mem1_test = compute_statis(p1_test, test_t)
    image = paramgraphics.mat_to_img(mem1_test.reshape((10*70,-1)), dim_input=(10,70), colorImg=colorImg, tile_shape=tile_shape, scale=scale)
    image.save(os.path.join(res_out, 'test1.png'), 'PNG')

    mem1_train = np.cov(mem1_train)
    mem2_train = np.cov(mem2_train)
    mem1_test = np.cov(mem1_test)
    mem2_test = np.cov(mem2_test)
    rawdata_train = np.cov(compute_statis(train_x, train_t))
    rawdata_test = np.cov(compute_statis(test_x, test_t))

    sio.savemat(os.path.join(res_out,'mem_cov.mat'), {'mem1_train' : mem1_train, 'mem2_train' : mem2_train, 'mem1_test' : mem1_test, 'mem2_test' : mem2_test, 'rawdata_train': rawdata_train, 'rawdata_test' : rawdata_test})
    
    
elif analysis_mode == 'classification':
    train_f = get_train_f(1000)
    test_f = get_test_f(1000)
    print train_f.shape
    print test_f.shape
    print train_t.shape
    print test_t.shape
    f = gzip.open(res_out+'/feature_target', 'wb')
    cPickle.dump([train_f, train_t, test_f, test_t], f)
    f.close()

elif analysis_mode == 'visualization_mem': 
    # visualization of memory slots directly
    print 'Visualizing...'
    num_vis = 70
    zero_h_g = np.zeros((num_vis, n_hiddens[0])).astype(np.float32)
    iden_prob = np.identity(num_vis).astype(np.float32)
    i_mem, mem = visualization_model_mem(zero_h_g, iden_prob)
    x_mean = i_mem
    tile_shape = (7, 10)
    image = paramgraphics.mat_to_img(x_mean.T, dim_input, colorImg=colorImg, tile_shape=tile_shape, scale=True)
    image.save(os.path.join(res_out, 'visualization.png'), 'PNG')
    image = paramgraphics.mat_to_img(x_mean.T, dim_input, colorImg=colorImg, tile_shape=tile_shape, scale=False)
    image.save(os.path.join(res_out, 'visualization_no_scale.png'), 'PNG')
    print x_mean.shape
    mem_activations = sio.loadmat(mem_activations_dir)
    test1 = mem_activations['mem1_test']
    print test1.shape
    print test1.sum(axis=1).min()
    print test1.sum(axis=1).max() # check
    # for each class select top-k slots
    k = 3
    probabilities = np.zeros((k, 10))
    indices = np.zeros((k, 10))
    for i in xrange(10):
        current_class = test1[i,:]
        #print current_class.shape
        indices[:,i] = (current_class.argsort()[-k:][::-1])
        indices[:,i] = (indices[:,i]).astype(np.int32)
        probabilities[:,i] = current_class[(indices[:,i]).astype(np.int32)]

    def print_(cov):
        for i in xrange(cov.shape[0]):
            for j in xrange(cov.shape[1]):
                print "%.3f " % cov[i][j],
            print '\n'

    print '##'
    print_(probabilities)
    print '##'
    print indices
    print '##'
    useful_image = x_mean[indices.flatten().astype(np.int32),:]
    useful_tile_shape = (k, 10)
    image = paramgraphics.mat_to_img(useful_image.T, dim_input, colorImg=colorImg, tile_shape=useful_tile_shape, scale=True)
    image.save(os.path.join(res_out, 'useful_visualization.png'), 'PNG')
    image = paramgraphics.mat_to_img(useful_image.T, dim_input, colorImg=colorImg, tile_shape=useful_tile_shape, scale=False)
    image.save(os.path.join(res_out, 'useful_visualization_no_scale.png'), 'PNG')

else:
    print 'Wrong analysis mode'
'''
elif analysis_mode == 'visualization_reco': 
    # visualization with memory, without memory and only memory
    print 'Visualizing...'
    num_vis = 20
    x_used = train_x[:num_vis,:]
    zero_h_g = np.zeros((num_vis, n_hiddens[0])).astype(np.float32)
    zero_h_m = zero_h_g
    i_with_mem, i_without_mem, i_mem, probs = visualization_model_reco(1,1,x_used, zero_h_g, zero_h_m)
    print (i_mem.var(axis=0)).mean()
    print probs.shape
    print (probs.var(axis=0)).mean()
    x_mean = np.vstack((x_used, i_with_mem, i_without_mem, i_mem))
    tile_shape = (4, num_vis)
    image = paramgraphics.mat_to_img(x_mean.T, dim_input, colorImg=colorImg, tile_shape=tile_shape, scale=True)
    image.save(os.path.join(res_out, 'visualization.png'), 'PNG')
    image = paramgraphics.mat_to_img(x_mean.T, dim_input, colorImg=colorImg, tile_shape=tile_shape, scale=False)
    image.save(os.path.join(res_out, 'visualization_no_scale.png'), 'PNG')


elif analysis_mode == 'visualization_gene': 
    # visualization with memory, without memory and only memory
    print 'Visualizing...'
    num_vis = 20
    input_z = np.random.normal(loc=0, scale=1, size=(num_vis, latent_size)).astype(np.float32)
    zero_h_g = np.ones((num_vis, n_hiddens[0])).astype(np.float32)
    zero_h_m = zero_h_g
    i_with_mem, i_without_mem, i_mem, probs = visualization_model_gene(input_z, zero_h_g, zero_h_m)
    print (i_mem.var(axis=0)).mean()
    print probs.shape
    print (probs.var(axis=0)).mean()
    x_mean = np.vstack((i_with_mem, i_without_mem, i_mem))
    tile_shape = (3, num_vis)
    image = paramgraphics.mat_to_img(x_mean.T, dim_input, colorImg=colorImg, tile_shape=tile_shape, scale=True)
    image.save(os.path.join(res_out, 'visualization.png'), 'PNG')
    image = paramgraphics.mat_to_img(x_mean.T, dim_input, colorImg=colorImg, tile_shape=tile_shape, scale=False)
    image.save(os.path.join(res_out, 'visualization_no_scale.png'), 'PNG')
'''
