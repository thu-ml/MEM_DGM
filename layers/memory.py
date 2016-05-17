import lasagne
from lasagne import init
from lasagne import nonlinearities

import theano.tensor as T
import theano
import numpy as np


__all__ = [
    "MemoryLayer",
    "SimpleCompositionLayer", # simple plus
    "LadderCompositionLayer", # ladder like
]


class MemoryLayer(lasagne.layers.Layer):
    '''
    Layer that represents memory mechanism

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape

    n_slots : int 
    	# of slots

    d_slots : int
    	dim of slots

    C : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the weights.
        These should be a matrix with shape ``(num_inputs, num_slots)``.

    M : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the weights.
        These should be a matrix with shape ``(num_inputs, num_slots)``.

    b : Theano shared variable, expression, numpy array, callable or ``None``
        Initial value, expression or initializer for the biases. If set to
        ``None``, the layer will have no biases. Otherwise, biases should be
        a 1D array with shape ``(num_slots,)``.

    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. If None
        is provided, the layer will be linear.
        

    References
    ----------
    '''
    def __init__(self, incoming, n_slots, d_slots, C=init.GlorotUniform(), M=init.Normal(),
                 b=init.Constant(0.), nonlinearity_final=nonlinearities.identity,
                 **kwargs):
    	super(MemoryLayer, self).__init__(incoming, **kwargs)
        
        self.nonlinearity_final = nonlinearity_final
        self.n_slots = n_slots
        self.d_slots = d_slots

        num_inputs = int(np.prod(self.input_shape[1:]))

        self.C = self.add_param(C, (num_inputs, n_slots), name="C") # controller
        self.M = self.add_param(M, (n_slots, d_slots), name="M") # memory slots
        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (n_slots,), name="b",
                                    regularizable=False)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.d_slots)

    def get_output_for(self, input, **kwargs):
        assert input.ndim == 2
        activation = T.dot(input, self.C)
        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        return self.nonlinearity_final(nonlinearities.sigmoid(activation).dot(self.M))

class SimpleCompositionLayer(lasagne.layers.MergeLayer):
    """
    Layers to composite information from inference and memory
    The motivation is as follows:
    h_g = f(x), x_pre = g(h_g); but g can't inverse f typically if h_g is invariant in some sense.
    x = g'(x_pre + h_m) is more reasonable.

    Parameters
    ----------
    x_pre, h_m : :class:`Layer` instances
       Iinformation from inference and memory respectively.

    W : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the weights.
        These should be a matrix with shape ``(num_inputs_g, num_units_g)``.

    b : Theano shared variable, expression, numpy array, callable or ``None``
        Initial value, expression or initializer for the biases. If set to
        ``None``, the layer will have no biases. Otherwise, biases should be
        a 1D array with shape ``(num_units_g,)``.

    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. If None
        is provided, the layer will be linear.

    References
    ----------
    """
    def __init__(self, x_pre, h_m, nonlinearity_final=nonlinearities.identity, **kwargs):

        super(SimpleCompositionLayer, self).__init__([x_pre, h_m], **kwargs)

        self.nonlinearity_final = nonlinearity_final

        #self.num_units = num_units

        #num_inputs = int(np.prod(self.input_shapes[0][1:]))

        #self.W = self.add_param(W, (num_inputs, num_units), name="W")
        #if b is None:
        #    self.b = None
        #else:
        #    self.b = self.add_param(b, (num_units,), name="b",
        #                            regularizable=False)

    def get_output_shape_for(self, input_shapes):
        return self.input_shapes[1]

    def get_output_for(self, input, **kwargs):
        x_pre, h_m = input
        return self.nonlinearity_final(h_m + x_pre)


class LadderCompositionLayer(lasagne.layers.MergeLayer):
    """
    Ladder network like composition layer    
    """
    def __init__(self, u_net, z_net,
                 nonlinearity=nonlinearities.sigmoid,
                 nonlinearity_final=nonlinearities.identity, **kwargs):
        super(LadderCompositionLayer, self).__init__([u_net, z_net], **kwargs)

        u_shp, z_shp = self.input_shapes


        if not u_shp[-1] == z_shp[-1]:
            raise ValueError("last dimension of u and z  must be equal"
                             " u was %s, z was %s" % (str(u_shp), str(z_shp)))
        self.num_inputs = z_shp[-1]
        self.nonlinearity = nonlinearity
        self.nonlinearity_final = nonlinearity_final
        constant = init.Constant
        self.a1 = self.add_param(constant(0.), (self.num_inputs,), name="a1")
        self.a2 = self.add_param(constant(1.), (self.num_inputs,), name="a2")
        self.a3 = self.add_param(constant(0.), (self.num_inputs,), name="a3")
        self.a4 = self.add_param(constant(0.), (self.num_inputs,), name="a4")

        self.c1 = self.add_param(constant(0.), (self.num_inputs,), name="c1")
        self.c2 = self.add_param(constant(1.), (self.num_inputs,), name="c2")
        self.c3 = self.add_param(constant(0.), (self.num_inputs,), name="c3")

        self.c4 = self.add_param(constant(0.), (self.num_inputs,), name="c4")

        self.b1 = self.add_param(constant(0.), (self.num_inputs,),
                                 name="b1", regularizable=False)

    def get_output_shape_for(self, input_shapes):
        output_shape = list(input_shapes[0])  # make a mutable copy
        return tuple(output_shape)

    def get_output_for(self, inputs, **kwargs):
        '''
        Exchange the order of the inputs, which is different with Ladder network.
        Same to change the initialization of the parameters.
        '''
        z_lat, u = inputs
        assert z_lat.ndim == 2

        sigval = self.c1 + self.c2*z_lat
        sigval += self.c3*u + self.c4*z_lat*u
        sigval = self.nonlinearity(sigval)
        z_est = self.a1 + self.a2 * z_lat + self.b1*sigval
        z_est += self.a3*u + self.a4*z_lat*u
        
        #mu = self.a1*self.nonlinearity(self.a2*u)
        return self.nonlinearity_final(z_est)