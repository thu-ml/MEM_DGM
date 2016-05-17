import lasagne
from lasagne import init
from lasagne.utils import floatX
from lasagne.random import get_rng

import theano.tensor as T
import theano
import numpy as np

class GlorotTanh(init.Initializer):
    """
    Initialize W given tanh activation
    """
    def __init__(self, factor=1.0):
        self.factor = factor

    def sample(self, shape):
        assert (len(shape) == 2)
        scale = self.factor*np.sqrt(6./(shape[0]+shape[1]))
        return floatX(get_rng().uniform(low=-scale, high=scale, size=shape))