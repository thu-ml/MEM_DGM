'''
Code for mmDGM
Author: Chongxuan Li (chongxuanli1991@gmail.com)
Version = '1.0'
'''

import numpy as np
import os, sys
import time
import cPickle, gzip
import scipy.io as sio

class Pegasos:
    """
    a simple implementation of pegasos for multiple classification
    
    pegasos_k: int
        size of the mini batch
        
    pegasos_T: int
        iteration times
        
    pegasos_lambda: float
        trade-off between weight decay and hinge-loss
    
    number_classes: int
        number of possible labels
    
    eta : float matrix [nc dimX]
        weight vectors learned by pegasos
    
    L : float, optional 
        using 0-L loss instead of 0-1 loss in SVM
        
    Check_gradient : binary, flag
        using numeric method to check gradient
        
    total_gradient : float same_size with pegasos
        for adagrad
    
    learning_rate : float
        global learning rate
    """
    
    def __init__(self, pegasos_k = 100, pegasos_T = 500, pegasos_lambda = 0.1, nc = 2, L = 1):
        self.pegasos_T = pegasos_T
        self.pegasos_k = pegasos_k
        self.pegasos_lambda = pegasos_lambda
        self.nc = nc
        self.L = L

    def init_H(self, A_t, y):
        grad = self.computeGradient(A_t, y)
        self.total_gradient += grad**2
        
    def init_param(self, dimX):
        self.eta = np.random.normal(0, 0.01,(self.nc, dimX))
        self.check_gradient = False
        self.check_objective = False
        self.learning_rate = 0.1
        self.total_gradient  = np.zeros((self.nc, dimX))
        
    def pegasos_optimize(self, X, Y, X_t, Y_t):
        # initialize weight as a zero vector
        [N, dimX] = X.shape
        self.init_param(dimX)
        # sample batches
        batches = np.arange(0, N, self.pegasos_k)
        if batches[-1] != N:
            batches = np.append(batches, N)
        #print batches
        
        if self.check_objective:
            self.fp = open("result",'w')
        
        for i in xrange(10):
            ii = i % (len(batches) - 2)
            minibatch = X[batches[ii]:batches[ii + 1]]
            label = Y[batches[ii]:batches[ii + 1]]
            self.init_H(minibatch.T, label)
        
        for j in xrange(self.pegasos_T):
            jj = j % (len(batches) - 2)
            minibatch = X[batches[jj]:batches[jj + 1]]
            label = Y[batches[jj]:batches[jj + 1]]
            
            self.pegasos_iter(minibatch.T, label, j + 1)
            
            # check the value of the objective funtion
            if self.check_objective:
                self.objective(X.T, Y, self.eta)
            
            if ((j+1) % 1000 == 0):
                print "Iteration: ", j + 1, " Testing Score: ", self.pegasos_score(X_t,Y_t)
                #" Training Score: ", self.pegasos_score(X,Y), 
                #print self.eta
            
        if self.check_objective:
            self.fp.close()
        
    def pegasos_iter(self, A_t, y, t):
        # check the gradient using numeric method
        if self.check_gradient:
            print 'Compute numgrad...'
            numgrad = self.numericGradient(self.objective, A_t, y, self.eta);
            
        grad = self.computeGradient(A_t, y)
        
        if False:
            eta_t = 1.0/(t * self.pegasos_lambda)
            self.eta -= eta_t * grad
        else:
            self.total_gradient += grad**2
            self.eta -= self.learning_rate * (grad / (1e-4 + np.sqrt(self.total_gradient)))
        
        if self.check_gradient:
            print 'grad: ', np.sum((grad)**2)
            print 'This relative ratio should be small: ', np.sum((grad-numgrad)**2)/np.sum((grad+numgrad)**2)
    
    def computeGradient(self, A_t, y):
        """
            computeGradient
        """
        
        [dimX, k] = A_t.shape
        
        # generate the 0-1 loss matrix
        l_y = np.ones((self.nc, k))
        l_y[y, xrange(k)] = 0
        l_y = l_y * self.L
        
        # compute the result
        """
            This line is the bottleneck of the algorithm 
        """
        multi_result = self.eta.dot(A_t)
        
        # generate the matrix whose columns are filled with only true label result 
        label_result = multi_result[y, xrange(k)] * np.ones((self.nc,k))
        
        # find the max label
        y_m = np.argmax(l_y - label_result + multi_result, axis = 0)
        
        # compute gradient
        grad = np.zeros(self.eta.shape)
        
        # a vectorization version of updating
        for cc in range(self.nc):
            grad[cc, :] += (A_t[:, y_m == cc]).sum(axis = 1)
            grad[cc, :] -= (A_t[:, y == cc]).sum(axis = 1)
        grad /= k;
        grad += self.eta * self.pegasos_lambda
        
        return grad
        
        
    def objective(self, data, y, eta_x):
        f = self.pegasos_lambda / 2 * np.sum(eta_x**2)
        
        [dimX, m] = data.shape
        
        # generate the 0-L loss matrix
        l_y = np.ones((self.nc, m))
        l_y[y, xrange(m)] = 0
        l_y = l_y * self.L

        # compute the result
        multi_result = eta_x.dot(data)

        # generate the matrix whose columns are filled with only true label result 
        label_result = multi_result[y, xrange(m)] * np.ones((self.nc,m))
        
        # compute the hinge loss
        hinge_loss = np.max(l_y - label_result + multi_result, axis = 0)
        self.fp.writelines(str(f)+' '+str(np.sum(hinge_loss) / m)+'\n') 

        f += np.sum(hinge_loss) / m
        return f
        
    def numericGradient(self, function, data, y, x):
        """
            The function that computes the numeric gradient
            'function' is real-valued function over 'x'
            'x' must be a matrix    
            'numgrad' has the same dimension with 'x'
        """
        
        EPS = 1e-4
        numgrad = np.zeros(x.shape)
        d1 = x.shape[0]
        d2 = x.shape[1]    
        
        for dd1 in xrange(d1):
            for dd2 in xrange(d2):
                tmp = np.zeros(x.shape)
                tmp[dd1,dd2] = EPS
                numgrad[dd1,dd2] = 0.5*(function(data, y, x+tmp)-function(data, y, x-tmp)) / EPS
        return numgrad
    
    def testNum(self,a,b,x):
        value = x[0,0]**2 + 3*x[0,0]*x[0,1]    
        return value
        
    def gradtestNum(self,a,b,x):
        grad = np.zeros(x.shape);
        grad[0,0] = 2*x[0,0] + 3*x[0,1]
        grad[0,1] = 3*x[0,0]
        return grad
        
    def get_eta(self):
        return self.eta
        
    def pegasos_score(self, X, Y):
        predict = np.argmax((self.eta.dot(X.T)), axis = 0)
        result = np.zeros(Y.shape[0])
        result[predict == Y] = 1
        return np.sum(result)/ Y.shape[0]
        
    def pegasos_score_compare(self, X, Y, eta):
        predict = np.argmax((self.eta.dot(X.T)), axis = 0)
        result = np.zeros(Y.shape[0])
        result[predict == Y] = 1
        print np.sum(result)/ Y.shape[0]
        
        predict = np.argmax((eta.dot(X.T)), axis = 0)
        result = np.zeros(Y.shape[0])
        result[predict == Y] = 1
        print np.sum(result)/ Y.shape[0]

if __name__ == "__main__":
    
    feature_file = sys.argv[1]

    np.random.seed(1234) # not fixed in the experiments reported in the paper.

    f = gzip.open(feature_file, 'rb')
    train_data,train_label,test_data,test_label = cPickle.load(f)
    print train_data.shape
    print train_label.shape
    print test_data.shape
    print test_label.shape
    print 'The test score means accuracy'
    nc = 10
    pegasos_batch = 100
    lam = 1E-4
    T = 1 / lam * 20
    T = int(T)
    param = dict()
    param['L'] = 1.0
    param['T'] = T

    p = Pegasos(pegasos_batch, param['T'] , lam, nc, param['L'])
    p.pegasos_optimize(train_data, train_label, test_data, test_label)
    print str(param), "Testing score: ", p.pegasos_score(test_data, test_label)
    with open('results/log.txt', 'a') as f:
        print >>f, str(param), "Testing score: ", p.pegasos_score(test_data, test_label)
