#encoding=utf-8
import numpy as np
import scipy.sparse as ssp
from sklearn.utils import column_or_1d
from sklearn.utils.extmath import (logsumexp, log_logistic, safe_sparse_dot,
                             squared_norm)
from sklearn.preprocessing import StandardScaler,LabelBinarizer
from sklearn.metrics import roc_auc_score,mean_squared_error,accuracy_score
from sklearn.base import BaseEstimator
from sklearn.utils.extmath import fast_dot
from scipy.sparse import issparse
from scipy import sparse
import cPickle
import copy
import h5py
import tables as tb
import theano
from theano import tensor as T


"""
Created on 2015-09-05
@author: qqian 

this is a Logistic Regression Classifier implementing google's FTRL-proximal(follow the regularized leader) training algorithm

"""


def categorical_probas_to_classes(p):
    return np.argmax(p, axis=1)

def store_sparse_mat(m, name, store='store.h5'):
    msg = "This code only works for csr matrices"
    assert(m.__class__ == sparse.csr.csr_matrix), msg
    with tb.openFile(store,'a') as f:
        for par in ('data', 'indices', 'indptr', 'shape'):
            full_name = '%s_%s' % (name, par)
            try:
              n = getattr(f.root, full_name)
              n._f_remove()
            except AttributeError:
              pass

            arr = np.array(getattr(m, par))
            atom = tb.Atom.from_dtype(arr.dtype)
            ds = f.createCArray(f.root, full_name, atom, arr.shape)
            ds[:] = arr

def load_sparse_mat(name, store='final.h5'):
    with tb.openFile(store) as f:
        pars = []
        for par in ('data', 'indices', 'indptr', 'shape'):
            pars.append(getattr(f.root, '%s_%s' % (name, par)).read())
    m = sparse.csr_matrix(tuple(pars[:3]), shape=pars[3])
    return m

def test_train_data(X,target,p=0.9):
    l =X.shape[0]
    to = (int)(l*p)
    X_train = X[0:to]
    target_train = target[0:to]
    X_test = X[to:]
    target_test = target[to:]
    return X_train,target_train,X_test,target_test


def slice_X(X, start=None, stop=None):
    if type(X) == list:
        if hasattr(start, '__len__'):
            return [x[start] for x in X]
        else:
            return [x[start:stop] for x in X]
    else:
        if hasattr(start, '__len__'):
            return X[start]
        else:
            return X[start:stop]

def make_batches(size, batch_size):
    nb_batch = int(np.ceil(size/float(batch_size)))
    return [(i*batch_size, min(size, (i+1)*batch_size)) for i in range(0, nb_batch)]


class WeightMixin():
    def save_sparse_weights(self):
        store_sparse_mat(ssp.hstack(self.n).tocsr(), name='n_%s'%self.name, store='ftrl.h5')
        store_sparse_mat(ssp.hstack(self.z).tocsr(), name='z_%s'%self.name, store='ftrl.h5')
        store_sparse_mat(ssp.hstack(self.w).tocsr(), name='w_%s'%self.name, store='ftrl.h5')

    def load_sparse_weights(self):
        self.n = load_sparse_mat(name='n_%s'%self.name, store='ftrl.h5')
        self.z = load_sparse_mat(name='z_%s'%self.name, store='ftrl.h5')
        self.w = load_sparse_mat(name='w_%s'%self.name, store='ftrl.h5')
        




class FtrlClassifier(BaseEstimator,WeightMixin):

    def get_fans(self,shape):
        fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
        fan_out = shape[1] if len(shape) == 2 else shape[0]
        return fan_in, fan_out
    def uniform(self,shape, scale=0.05):
        return np.random.uniform(low=-scale, high=scale, size=shape)

    def glorot_uniform(self,shape):
        fan_in, fan_out = self.get_fans(shape)
        s = np.sqrt(6. / (fan_in + fan_out))
        return self.uniform(shape, s)

    def binary_entropy(self,p, y):
        loss =-(np.array(y) * np.log(p) + (1.0 - np.array(y)) * np.log(1.0 - np.array(p)))
        return np.mean(loss)

    def get_weights(self):
        return self.w

    def get_z(self):
        return self.z
        
    def get_params(self, deep=True):
        return self.params
 
    def set_params(self, **params):
        self.params.update(params)
        return self

    def load_weights(self):
        f = open("best_ftrl_%s.mdl"%self.name,"rb")
        best_weights = cPickle.load(f)
        self.z = best_weights[0]
        self.n = best_weights[1]
        self.w = best_weights[2]
        f.close()

    def __init__(self,alpha=0.005, beta=1, l1=0.0, l2=0.0,nb_epoch=20,batch_size=128,batch_normalization=False,early_stop_rounds=None,use_glorot=False,drop_out=0.0,eval_function=roc_auc_score,validation_set=None,validation_split = None,name="default"):
        """ Get probability estimation on x

            INPUT:
                alpha: float, alpha
                beta: float, beta
                l1: float, l1 penalty
                l2: float, l2 penalty
                nb_epoch: int, number of epochs to train
                batch_size: int,mini batch size
                early_stop_rounds:int, early_stop_rounds only applys when validation_set is used
                use_glorot: boolean, use glorot_uniform to initialize weights default is false
        """
        # parameters for training
        self.alpha = alpha
        self.beta = beta
        self.l1 = l1
        self.l2 = l2
        self.nb_epoch = nb_epoch
        self.iterations =0
        self.batch_size = batch_size
        self.eval_function= eval_function
        self.name=name
        self.drop_out =1-drop_out
        self.batch_normalization = batch_normalization
        

        # feature related parameters
        # model
        # n: squared sum of past gradients
        # z: weights
        # w: lazy weights
        # g: graident
        self.z = None
        self.n = None
        self.w = None
        self.g = 0.
        # initialize other params
        self.loss = 0.
        self.count = 0
        self.early_stop_rounds = early_stop_rounds
        self.use_glorot= use_glorot
        self.validation_set = validation_set
        self.validation_split = validation_split
    


    def _validate_y(self, y):
        y = column_or_1d(y, warn=True)
        self.classes_, y = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        return y


    def fit(self,X,y,verbose=1,shuffle=True,validation_set=None,validation_split = None,sample_weight=None):
        if validation_set==None:
            validation_set = self.validation_set
        if validation_split==None:
            validation_split = self.validation_split
        if validation_split!=None:
            X,y,xt,yt=test_train_data(X,y,p=validation_split)
            validation_set=[]
            validation_set.append(xt)
            validation_set.append(yt)

        if self.batch_normalization:
            scaler = StandardScaler(with_mean=(not issparse(X)))
        # if sample_weight:
        #     sample_weight = np.zeros(y.shape[0])+1

        y = self._validate_y(y)
        name = self.name
        eval_function = self.eval_function
        if self.z == None and self.n == None and self.w ==None:
            
            self.input_size = X.shape[1]
            self.n = np.zeros(self.input_size)
            
            if self.use_glorot:
                self.w = self.glorot_uniform((self.input_size,))
                self.z = self.glorot_uniform((self.input_size,))
            else:
                self.w = np.zeros(self.input_size)
                self.z = np.zeros(self.input_size)

        batch_size = self.batch_size
        preds = []
        sample_size = X.shape[0]
        index_array = np.arange(sample_size)
        early_stop_rounds=self.early_stop_rounds
        best_auc = -np.inf
        count=0
        for epoch in range(self.nb_epoch):
            if shuffle:
                np.random.shuffle(index_array)
            batches = make_batches(sample_size, batch_size)

            for batch_index, (batch_start, batch_end) in enumerate(batches):
                batch_ids = index_array[batch_start:batch_end]
                ins_batch = slice_X(X, batch_ids)

                if self.batch_normalization:
                    ins_batch = scaler.fit_transform(ins_batch)

                outs = self.predict(ins_batch)
                if type(outs) != list:
                    outs = [outs]
                target = y[batch_ids]
                # self.update(ins_batch, outs, target)
                if sample_weight!=None:
                    self.update(ins_batch, outs, target,sample_weight[batch_ids])
                else:
                    self.update(ins_batch, outs, target)

            if early_stop_rounds and validation_set:
                tp = self.predict(validation_set[0])
                validation_auc = eval_function(validation_set[1],tp)
                if best_auc<validation_auc:
                    count=0
                    best_auc = validation_auc
                    f = open("best_ftrl_%s.mdl"%name,"wb")
                    cPickle.dump([self.z,self.n,self.w],f)
                    f.close()
                
                if count ==early_stop_rounds:
                    validation_auc = eval_function( validation_set[1],self.predict(validation_set[0]))
                    count=0
                    if best_auc>validation_auc:
                        break
                else:
                    count+=1


            if verbose>=1:
                self.loss = self.binary_entropy(self.predict(X), y)
                if validation_set:
                    validation_loss = self.binary_entropy(self.predict(validation_set[0]), validation_set[1])
                    validation_auc = eval_function(validation_set[1],self.predict(validation_set[0]))
                    print('eoch %s\tcurrent_loss: %f\tvalid_loss: %f\tvalid_auc: %f'%(epoch,self.loss,validation_loss,validation_auc))
                else:
                    print('eoch %s\tcurrent loss: %f'%(epoch,self.loss))
        if early_stop_rounds and validation_set:
            f = open("best_ftrl_%s.mdl"%name,"rb")
            best_weights = cPickle.load(f)
            self.z = best_weights[0]
            self.n = best_weights[1]
            self.w = best_weights[2]
            f.close()
           


    def predict(self,X):
        preds = []
        preds = self._predict(X)
        return self.sigmoid(preds)

    def predict_proba(self,X):

        return self._predict_proba_lr(X)


    def _predict_proba_lr(self, X):
        """Probability estimation for OvR logistic regression.

        Positive class probabilities are computed as
        1. / (1. + np.exp(-self.decision_function(X)));
        multiclass is handled by normalizing that over all classes.
        """
        self.decision_function = self._predict
        prob = self.decision_function(X)
        prob *= -1
        np.exp(prob, prob)
        prob += 1
        np.reciprocal(prob, prob)
        if len(prob.shape) == 1:
            return np.vstack([1 - prob, prob]).T
        else:
            # OvR normalization, like LibLinear's predict_probability
            prob /= prob.sum(axis=1).reshape((prob.shape[0], -1))
            return prob



    def _predict(self, x):
        ''' Get probability estimation on x

            INPUT:
                x: features

            OUTPUT:
                probability of p(y = 1 | x; W)
        '''
        # parameters
        w = self.get_weights()
        # wTx is the inner product of w and x'

        wTx = safe_sparse_dot(x,w, dense_output=False)
  
        # cache the current w for update stage
        self.w = w
        # bounded sigmoid function, this is the probability estimation
        p = wTx
        return p


    def sigmoid(self,inX):  
        return 1.0 / (1 + np.exp(-inX))


    def bounded_sigmoid(self,inX):  
        return 1. / (1. + np.exp(-np.max(np.min(inX, 35.), -35.)))
    


    def update(self, x, p, y,sample_weight=None):
        ''' Update model using x, p, y

            INPUT:
                x: inputs
                p: click probability prediction of our model
                y: target

            MODIFIES:
                self.n: increase by squared gradient
                self.z: weights
        '''
        # gradient under logloss
        # use drop out
        mask = np.random.binomial(1, self.drop_out, size=y.shape)
        y = y*mask 
        diff = p - y
        if sample_weight!=None:
            diff = sample_weight*diff

        self.g = safe_sparse_dot(diff,x).ravel()

        sigma = (np.sqrt(self.n + self.g * self.g) - np.sqrt(self.n)) / self.alpha
        sigma = sigma.ravel()
        self.z += self.g - sigma * self.w
        self.n += self.g * self.g
        self.iterations +=1
        w = (np.sign(self.z) * self.l1 - self.z) / ((self.beta + np.sqrt(self.n)) / self.alpha + self.l2)
        idx_0 = np.abs(self.z) <= self.l1
        w[idx_0]=0
        self.w = w

class SGDClassifier(BaseEstimator):

    def get_fans(self,shape):
        fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
        fan_out = shape[1] if len(shape) == 2 else shape[0]
        return fan_in, fan_out
    def uniform(self,shape, scale=0.05):
        return np.random.uniform(low=-scale, high=scale, size=shape)

    def glorot_uniform(self,shape):
        fan_in, fan_out = self.get_fans(shape)
        s = np.sqrt(6. / (fan_in + fan_out))
        return self.uniform(shape, s)

    def binary_entropy(self,p, y):
        loss =-(np.array(y) * np.log(p) + (1.0 - np.array(y)) * np.log(1.0 - np.array(p)))
        return np.mean(loss)

    def get_weights(self):
        return self.w


    def get_z(self):
        return self.z


    def __init__(self,alpha=0.005, beta=1, l2=0.0,nb_epoch=20,batch_size=128,early_stop_rounds=None,use_glorot=False,eval_function=roc_auc_score,validation_set=None,validation_split = None,name="default"):
        """ Get probability estimation on x

            INPUT:
                alpha: float, alpha
                beta: float, beta
                l2: float, l2 penalty
                nb_epoch: int, number of epochs to train
                batch_size: int,mini batch size
                early_stop_rounds:int, early_stop_rounds only applys when validation_set is used
                use_glorot: boolean, use glorot_uniform to initialize weights default is false
        """
        # parameters for training
        self.alpha = alpha
        self.beta = beta
        self.l2 = l2
        self.nb_epoch = nb_epoch
        self.iterations =0
        self.batch_size = batch_size
        self.eval_function= eval_function
        self.name=name
 

        # feature related parameters
        # model
        # n: squared sum of past gradients
        # z: weights
        # w: lazy weights
        # g: gradient
        self.z = None
        self.n = None
        self.w = None
        self.g = 0.
        # initialize other params
        self.loss = 0.
        self.count = 0
        self.early_stop_rounds = early_stop_rounds
        self.use_glorot= use_glorot
        self.validation_set = validation_set
        self.validation_split = validation_split
    


    def _validate_y(self, y):
        y = column_or_1d(y, warn=True)
        self.classes_, y = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        return y


    def fit(self,X,y,verbose=1,shuffle=True,validation_set=None,validation_split = None):
        if validation_set==None:
            validation_set = self.validation_set
        if validation_split==None:
            validation_split = self.validation_split
        if validation_split!=None:
            X,y,xt,yt=test_train_data(X,y,p=validation_split)
            validation_set=[]
            validation_set.append(xt)
            validation_set.append(yt)
        y = self._validate_y(y)
        name = self.name
        eval_function = self.eval_function
        if self.n == None and self.w ==None:
            
            self.input_size = X.shape[1]
            self.n = np.zeros(self.input_size)
            
            if self.use_glorot:
                self.w = self.glorot_uniform((self.input_size,))
            else:
                self.w = np.zeros(self.input_size)


        batch_size = self.batch_size
        preds = []
        sample_size = X.shape[0]
        index_array = np.arange(sample_size)
        early_stop_rounds=self.early_stop_rounds
        best_auc = 0
        count=0
        for epoch in range(self.nb_epoch):
            if shuffle:
                np.random.shuffle(index_array)
            batches = make_batches(sample_size, batch_size)

            for batch_index, (batch_start, batch_end) in enumerate(batches):
                batch_ids = index_array[batch_start:batch_end]
                ins_batch = slice_X(X, batch_ids)
                outs = self.predict(ins_batch)
                if type(outs) != list:
                    outs = [outs]
                target = y[batch_ids]
                self.update(ins_batch, outs, target)

            if early_stop_rounds and validation_set:
                tp = self.predict(validation_set[0])
                #print tp,validation_set[0],validation_set[1]  
                validation_auc = eval_function(validation_set[1],tp)
                if best_auc<validation_auc:
                    count=0
                    best_auc = validation_auc
                    f = open("best_sgd_%s.mdl"%name,"wb")
                    cPickle.dump([self.z,self.n,self.w],f)
                    f.close()
                
                if count ==early_stop_rounds:
                    validation_auc = eval_function( validation_set[1],self.predict(validation_set[0]))
                    count=0
                    if best_auc>validation_auc:
                        break
                else:
                    count+=1


            if verbose>=1:
                self.loss = self.binary_entropy(self.predict(X), y)
                if validation_set:
                    validation_loss = self.binary_entropy(self.predict(validation_set[0]), validation_set[1])
                    validation_auc = eval_function(validation_set[1],self.predict(validation_set[0]))
                    print('eoch %s\tcurrent_loss: %f\tvalid_loss: %f\tvalid_auc: %f'%(epoch,self.loss,validation_loss,validation_auc))
                else:
                    print('eoch %s\tcurrent loss: %f'%(epoch,self.loss))
        if early_stop_rounds and validation_set:
            f = open("best_sgd_%s.mdl"%name,"rb")
            best_weights = cPickle.load(f)
            self.z = best_weights[0]
            self.n = best_weights[1]
            self.w = best_weights[2]
            f.close()
           


    def predict(self,X):
        preds = []
        preds = self._predict(X)
        return self.sigmoid(preds)

    def predict_proba(self,X):
        return self._predict_proba_lr(X)


    def _predict_proba_lr(self, X):
        """Probability estimation for OvR logistic regression.

        Positive class probabilities are computed as
        1. / (1. + np.exp(-self.decision_function(X)));
        multiclass is handled by normalizing that over all classes.
        """
        self.decision_function = self._predict
        prob = self.decision_function(X)
        prob *= -1
        np.exp(prob, prob)
        prob += 1
        np.reciprocal(prob, prob)
        if len(prob.shape) == 1:
            return np.vstack([1 - prob, prob]).T
        else:
            # OvR normalization, like LibLinear's predict_probability
            prob /= prob.sum(axis=1).reshape((prob.shape[0], -1))
            return prob



    def _predict(self, x):
        ''' Get probability estimation on x

            INPUT:
                x: features

            OUTPUT:
                probability of p(y = 1 | x; W)
        '''
        w = self.w
        # wTx is the inner product of w and x'
        wTx = safe_sparse_dot(x,w, dense_output=False)
  
        # cache the current w for update stage
        self.w = w

        p = wTx

        return p


    def sigmoid(self,inX):  
        return 1.0 / (1 + np.exp(-inX))


    def bounded_sigmoid(self,inX):  
        return 1. / (1. + np.exp(-np.max(np.min(inX, 35.), -35.)))
    


    def update(self, x, p, y,sample_weight=None):
        ''' Update model using x, p, y

            INPUT:
                x: inputs
                p: click probability prediction of our model
                y: target

            MODIFIES:
                self.n: increase by squared gradient
                self.z: weights
        '''
        # gradient under logloss
        self.g = safe_sparse_dot((p - y),x).ravel()

        self.n += self.g * self.g
        self.iterations +=1
        self.w -=self.alpha/(self.beta+np.sqrt(self.n)) * (self.g - self.l2*self.w)


class FtrlRegressor(BaseEstimator,WeightMixin):
    
    def get_params(self, deep=True):
        return self.params
    
    def set_params(self, **params):
        self.params.update(params)
        return self
        
    def get_fans(self,shape):
        fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
        fan_out = shape[1] if len(shape) == 2 else shape[0]
        return fan_in, fan_out
    def uniform(self,shape, scale=0.05):
        return np.random.uniform(low=-scale, high=scale, size=shape)

    def glorot_uniform(self,shape):
        fan_in, fan_out = self.get_fans(shape)
        s = np.sqrt(6. / (fan_in + fan_out))
        return self.uniform(shape, s)

    def binary_entropy(self,p, y):
        loss =-(np.array(y) * np.log(p) + (1.0 - np.array(y)) * np.log(1.0 - np.array(p)))
        return np.mean(loss)


    def get_weights(self):
        return self.w


    def get_z(self):
        return self.z


    def __init__(self,alpha=0.005, beta=1, l1=0.0, l2=0.0,nb_epoch=20,batch_size=128,batch_normalization=False,early_stop_rounds=None,drop_out=0.0,use_glorot=False,eval_function=mean_squared_error,validation_set=None,validation_split = None,name="regressor",**params):
        """ Get probability estimation on x

            INPUT:
                alpha: float, alpha
                beta: float, beta
                l1: float, l1 penalty
                l2: float, l2 penalty
                nb_epoch: int, number of epochs to train
                batch_size: int,mini batch size
                early_stop_rounds:int, early_stop_rounds only applys when validation_set is used
                use_glorot: boolean, use glorot_uniform to initialize weights default is false
        """
        # parameters for training
        self.alpha = alpha
        self.beta = beta
        self.l1 = l1
        self.l2 = l2
        self.drop_out = 1-drop_out
        self.nb_epoch = nb_epoch
        self.iterations =0
        self.batch_size = batch_size
        self.eval_function= eval_function
        self.name=name
        self.batch_normalization = batch_normalization
        self.params = params
        
        # feature related parameters
        # model
        # n: squared sum of past gradients
        # z: weights
        # w: lazy weights
        # g: graient
        self.z = None
        self.n = None
        self.w = None
        self.g = 0.
        # initialize other params
        self.loss = 0.
        self.count = 0
        self.early_stop_rounds = early_stop_rounds
        self.use_glorot= use_glorot
        self.validation_set = validation_set
        self.validation_split = validation_split
    


    def _validate_y(self, y):
        y = column_or_1d(y, warn=True)
        self.classes_, y = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        return y


    def fit(self,X,y,verbose=1,shuffle=True,validation_set=None,validation_split = None,sample_weight=None):
        if validation_set==None:
            validation_set = self.validation_set
        if validation_split==None:
            validation_split = self.validation_split
        if validation_split!=None:
            X,y,xt,yt=test_train_data(X,y,p=validation_split)
            validation_set=[]
            validation_set.append(xt)
            validation_set.append(yt)

        if self.batch_normalization:
            scaler = StandardScaler(with_mean=(not issparse(X)))

        self.sample_weight=sample_weight
        # y = self._validate_y(y)
        name = self.name
        eval_function = self.eval_function
        if self.z == None and self.n == None and self.w ==None:
            self.input_size = X.shape[1]
            #print self.input_size
            self.n = np.zeros(self.input_size)
            
            if self.use_glorot:
                self.w = self.glorot_uniform((self.input_size,))
                self.z = self.glorot_uniform((self.input_size,))
            else:
                self.w = np.zeros(self.input_size)
                self.z = np.zeros(self.input_size)


        batch_size = self.batch_size
        preds = []
        sample_size = X.shape[0]
        index_array = np.arange(sample_size)
        early_stop_rounds=self.early_stop_rounds
        best_auc = np.inf
        count=0
        for epoch in range(self.nb_epoch):
            if shuffle:
                np.random.shuffle(index_array)
            batches = make_batches(sample_size, batch_size)

            for batch_index, (batch_start, batch_end) in enumerate(batches):
                batch_ids = index_array[batch_start:batch_end]
                ins_batch = slice_X(X, batch_ids)
                if self.batch_normalization:
                    ins_batch = scaler.fit_transform(ins_batch)

                outs = self.predict(ins_batch)
                if type(outs) != list:
                    outs = [outs]
                target = y[batch_ids]
                self.update(ins_batch, outs, target,sample_weight)

            if early_stop_rounds and validation_set:
                tp = self.predict(validation_set[0])
                #print tp,validation_set[0],validation_set[1]  
                validation_auc = eval_function(validation_set[1],tp)
                if best_auc>validation_auc:
                    count=0
                    best_auc = validation_auc
                    f = open("best_ftrl_%s.mdl"%name,"wb")
                    cPickle.dump([self.z,self.n,self.w],f)
                    f.close()
                
                if count ==early_stop_rounds:
                    validation_auc = eval_function( validation_set[1],self.predict(validation_set[0]))
                    count=0
                    if best_auc<validation_auc:
                        break
                else:
                    count+=1
            if verbose>=1:
                self.loss = self.eval_function(y,self.predict(X))
                if validation_set:
                    validation_loss = self.eval_function(validation_set[1],self.predict(validation_set[0]))
                    # validation_auc = eval_function(validation_set[1],self.predict(validation_set[0]))
                    print 'eoch %s\tcurrent_loss: %f\tvalid_loss: %f\t'%(epoch,self.loss,validation_loss)
                else:
                    print('eoch %s\tcurrent loss: %f'%(epoch,self.loss))
        if early_stop_rounds and validation_set:
            f = open("best_ftrl_%s.mdl"%name,"rb")
            best_weights = cPickle.load(f)
            self.z = best_weights[0]
            self.n = best_weights[1]
            self.w = best_weights[2]
            f.close()
        self.coef_ = self.w


    def load_weights(self):
        f = open("best_ftrl_%s.mdl"%self.name,"rb")
        best_weights = cPickle.load(f)
        self.z = best_weights[0]
        self.n = best_weights[1]
        self.w = best_weights[2]
        f.close()

    def predict(self,X):
        preds = []
        preds = self._predict(X)
        return preds

    def predict_proba(self,X):
        #return self.predict(X)
        return self._predict_proba_lr(X)


    def _predict_proba_lr(self, X):
        """Probability estimation for OvR logistic regression.

        Positive class probabilities are computed as
        1. / (1. + np.exp(-self.decision_function(X)));
        multiclass is handled by normalizing that over all classes.
        """
        self.decision_function = self._predict
        prob = self.decision_function(X)
        prob *= -1
        np.exp(prob, prob)
        prob += 1
        np.reciprocal(prob, prob)
        if len(prob.shape) == 1:
            return np.vstack([1 - prob, prob]).T
        else:
            # OvR normalization, like LibLinear's predict_probability
            prob /= prob.sum(axis=1).reshape((prob.shape[0], -1))
            return prob



    def _predict(self, x):
        ''' Get probability estimation on x

            INPUT:
                x: features

            OUTPUT:
                probability of p(y = 1 | x; W)
        '''
        # parameters
        w = self.get_weights()
        # wTx is the inner product of w and x'
        # if issparse(x):
        #     wTx = x.dot(w)#[0]
        #     #wTx = x*w
        #     #print wTx.shape
        # else:
        #     wTx = np.dot(x,w)
        wTx = safe_sparse_dot(x,w, dense_output=False)
  
        # cache the current w for update stage
        self.w = w
        # bounded sigmoid function, this is the probability estimation
        p = wTx
        #p = self.sigmoid(wTx)
        return p


    def sigmoid(self,inX):  
        return 1.0 / (1 + np.exp(-inX))


    def bounded_sigmoid(self,inX):  
        return 1. / (1. + np.exp(-np.max(np.min(inX, 35.), -35.)))
    


    def update(self, x, p, y,sample_weight=None):
        ''' Update model using x, p, y

            INPUT:
                x: inputs
                p: click probability prediction of our model
                y: target

            MODIFIES:
                self.n: increase by squared gradient
                self.z: weights
        '''
        # gradient under logloss
        #if input is sparse
        # if issparse(x):
        #     self.g = ((p-y)*x).ravel()
        # else:
        #     self.g = np.dot((p - y),x).ravel()
        # self.g = fast_dot((p - y),x).ravel()

        
        mask = np.random.binomial(1, self.drop_out, size=y.shape)
        # print 'y before',y.shape
        # print 'mask',mask.shape
        y = y*mask 
        # print 'y after',y.shape
        diff = p - y
        if sample_weight!=None:
            diff = sample_weight*diff
        
        self.g = safe_sparse_dot(diff,x).ravel()
        
        
        sigma = (np.sqrt(self.n + self.g * self.g) - np.sqrt(self.n)) / self.alpha
        sigma = sigma.ravel()

        
        

        # print sigma
        # print self.g
        # print self.z
        # print (sigma * self.w).shape
        # print 'self.g',self.g.shape
        # print 'sigma',sigma.shape
        # print 'self.w',self.w.shape
        # print "self.z",self.z.shape
        self.z += self.g - sigma * self.w
        self.n += self.g * self.g
        # self.iterations +=1
        w = (np.sign(self.z) * self.l1 - self.z) / ((self.beta + np.sqrt(self.n)) / self.alpha + self.l2)
        idx_0 = np.abs(self.z) <= self.l1
        w[idx_0]=0
        self.w = w



class SGDRegressor(BaseEstimator):

    def get_fans(self,shape):
        fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
        fan_out = shape[1] if len(shape) == 2 else shape[0]
        return fan_in, fan_out
    def uniform(self,shape, scale=0.05):
        return np.random.uniform(low=-scale, high=scale, size=shape)

    def glorot_uniform(self,shape):
        fan_in, fan_out = self.get_fans(shape)
        s = np.sqrt(6. / (fan_in + fan_out))
        return self.uniform(shape, s)

    def binary_entropy(self,p, y):
        loss =-(np.array(y) * np.log(p) + (1.0 - np.array(y)) * np.log(1.0 - np.array(p)))
        #loss_out = np.zeros(len(y))
        return np.mean(loss)

    def get_weights(self):
        return self.w


    def get_z(self):
        return self.z


    def __init__(self,alpha=0.005, momentum=0.9,l1=0.0, l2=0.0,drop_out=0.0,decay=1e-6,nesterov=False,nb_epoch=20,batch_size=128,early_stop_rounds=None,use_glorot=False,eval_function=mean_squared_error,validation_set=None,validation_split = None,name="regressor"):
        """ Get probability estimation on x

            INPUT:
                alpha: float, alpha
                beta: float, beta
                l2: float, l2 penalty
                nb_epoch: int, number of epochs to train
                batch_size: int,mini batch size
                early_stop_rounds:int, early_stop_rounds only applys when validation_set is used
                use_glorot: boolean, use glorot_uniform to initialize weights default is false
        """
        # parameters for training
        self.alpha = alpha
        self.momentum = momentum
        self.l1 =l1
        self.l2 = l2
        self.decay = decay
        self.nesterov = nesterov
        self.nb_epoch = nb_epoch
        self.iterations =0
        self.batch_size = batch_size
        self.eval_function= eval_function
        self.name=name
        self.drop_out =1-drop_out
        self.delta_w = 0
        # feature related parameters
        # model
        # n: squared sum of past gradients
        # z: weights
        # w: lazy weights
        # g: gradient
        self.z = None
        self.n = None
        self.w = None
        self.g = 0.
        # initialize other params
        self.loss = 0.
        self.count = 0
        self.early_stop_rounds = early_stop_rounds
        self.use_glorot= use_glorot
        self.validation_set = validation_set
        self.validation_split = validation_split
    


    def _validate_y(self, y):
        y = column_or_1d(y, warn=True)
        self.classes_, y = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        return y


    def fit(self,X,y,verbose=1,shuffle=True,validation_set=None,validation_split = None):
        if validation_set==None:
            validation_set = self.validation_set
        if validation_split==None:
            validation_split = self.validation_split
        if validation_split!=None:
            X,y,xt,yt=test_train_data(X,y,p=validation_split)
            validation_set=[]
            validation_set.append(xt)
            validation_set.append(yt)
        # y = self._validate_y(y)
        
        name = self.name
        eval_function = self.eval_function
        if self.n == None and self.w ==None:
            self.input_size = X.shape[1]
            #print self.input_size
            self.n = np.zeros(self.input_size)
            
            if self.use_glorot:
                self.w = self.glorot_uniform((self.input_size,))
            else:
                self.w = np.zeros(self.input_size)
                
                
        self.velocity = np.zeros(self.w.shape)

        batch_size = self.batch_size
        preds = []
        sample_size = X.shape[0]
        index_array = np.arange(sample_size)
        early_stop_rounds=self.early_stop_rounds
        best_auc = np.inf
        count=0
        for epoch in range(self.nb_epoch):
            if shuffle:
                np.random.shuffle(index_array)
            batches = make_batches(sample_size, batch_size)
            self.lr = self.alpha * (1.0 / (1.0 + self.decay * self.iterations))
            for batch_index, (batch_start, batch_end) in enumerate(batches):
                batch_ids = index_array[batch_start:batch_end]
                ins_batch = slice_X(X, batch_ids)
                outs = self.predict(ins_batch)

                if type(outs) != list:
                    outs = [outs]
                target = y[batch_ids]
                self.update(ins_batch, outs, target)
            self.iterations +=1
            if early_stop_rounds and validation_set:
                tp = self.predict(validation_set[0])
                #print tp,validation_set[0],validation_set[1]  
                validation_auc = eval_function(validation_set[1],tp)
                if best_auc>validation_auc:
                    count=0
                    best_auc = validation_auc
                    f = open("best_sgd_r_%s.mdl"%name,"wb")
                    cPickle.dump([self.z,self.n,self.w],f)
                    f.close()
                
                if count ==early_stop_rounds:
                    validation_auc = eval_function( validation_set[1],self.predict(validation_set[0]))
                    count=0
                    if best_auc<validation_auc:
                        break
                else:
                    count+=1

            if verbose>=1:
                self.loss = self.eval_function(y, self.predict(X))
                if validation_set:
                    validation_loss = self.eval_function(self.predict(validation_set[0]), validation_set[1])
                    # validation_auc = eval_function(validation_set[1],self.predict(validation_set[0]))
                    print('eoch %s\tcurrent_loss: %f\tvalid_loss: %f\t'%(epoch,self.loss,validation_loss))
                else:
                    print('eoch %s\tcurrent loss: %f'%(epoch,self.loss))
        if early_stop_rounds and validation_set:
            f = open("best_sgd_r_%s.mdl"%name,"rb")
            best_weights = cPickle.load(f)
            self.z = best_weights[0]
            self.n = best_weights[1]
            self.w = best_weights[2]
            f.close()
           


    def predict(self,X):
        preds = []
        preds = self._predict(X)
        return preds

    def predict_proba(self,X):
        return self._predict_proba_lr(X)


    def _predict_proba_lr(self, X):
        """Probability estimation for OvR logistic regression.

        Positive class probabilities are computed as
        1. / (1. + np.exp(-self.decision_function(X)));
        multiclass is handled by normalizing that over all classes.
        """
        self.decision_function = self._predict
        prob = self.decision_function(X)
        prob *= -1
        np.exp(prob, prob)
        prob += 1
        np.reciprocal(prob, prob)
        if len(prob.shape) == 1:
            return np.vstack([1 - prob, prob]).T
        else:
            # OvR normalization, like LibLinear's predict_probability
            prob /= prob.sum(axis=1).reshape((prob.shape[0], -1))
            return prob



    def _predict(self, x):
        ''' Get probability estimation on x

            INPUT:
                x: features

            OUTPUT:
                probability of p(y = 1 | x; W)
        '''

        w = self.w
        # print x.shape
        # wTx is the inner product of w and x'
        wTx = safe_sparse_dot(x,w, dense_output=True)
        
        # cache the current w for update stage
        self.w = w

        p = wTx

        return p


    def sigmoid(self,inX):  
        return 1.0 / (1 + np.exp(-inX))


    def bounded_sigmoid(self,inX):  
        return 1. / (1. + np.exp(-np.max(np.min(inX, 35.), -35.)))
    


    def update(self, x, p, y,sample_weight=None):
        ''' Update model using x, p, y

            INPUT:
                x: inputs
                p: click probability prediction of our model
                y: target

            MODIFIES:
                self.n: increase by squared gradient
                self.z: weights
        '''

        mask = np.random.binomial(1, self.drop_out, size=y.shape)
        y = y*mask 
        # gradient under logloss
        self.g = safe_sparse_dot((y - p),x).ravel()

        # print (y - p).shape

        # self.w -=self.alpha/(self.beta+np.sqrt(self.n)) * (self.g - self.l2*self.w)
        # print self.alpha*self.g
        self.velocity = self.lr*self.g+self.momentum*self.velocity

        # temp = copy.deepcopy(self.w)
        if self.nesterov:
            self.w +=self.momentum*self.velocity + self.lr * self.g#-0.5*self.l1*np.sign(self.w)-self.l2* np.sqrt(np.dot(self.w.ravel(),self.w.ravel()))
        else:
            self.w +=self.velocity#-0.5*self.l1*np.sign(self.w)-self.l2* np.sqrt(np.dot(self.w.ravel(),self.w.ravel()))
        # print 'delta_w',self.delta_w
        # self.delta_w = self.w-temp



class FtrlSoftmaxClassifier(BaseEstimator,WeightMixin):

    def get_fans(self,shape):
        fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
        fan_out = shape[1] if len(shape) == 2 else shape[0]
        return fan_in, fan_out
    def uniform(self,shape, scale=0.05):
        return np.random.uniform(low=-scale, high=scale, size=shape)

    def glorot_uniform(self,shape):
        fan_in, fan_out = self.get_fans(shape)
        s = np.sqrt(6. / (fan_in + fan_out))
        return self.uniform(shape, s)

    def binary_entropy(self,p, y):
        loss =-(np.array(y) * np.log(p) + (1.0 - np.array(y)) * np.log(1.0 - np.array(p)))
        return np.mean(loss)

    def categorical_entropy(self,p, y):
        mlogloss=0.0
        for n in range(y.shape[1]):
            loss_n =-(np.array(y)[:,n] * np.log(p)[:,n] + (1.0 - np.array(y)[:,n]) * np.log(1.0 - np.array(p)[:,n]))
            loss_n = np.mean(loss_n)
            mlogloss+=loss_n
        return mlogloss



    def get_weights(self):
        return self.w

    def get_z(self):
        return self.z

    
    def __init__(self,alpha=0.005, beta=1, l1=0.0, l2=0.0,nb_epoch=20,batch_size=128,batch_normalization=False,early_stop_rounds=None,use_glorot=False,drop_out=0.0,eval_function=accuracy_score,validation_set=None,validation_split = None,name="default"):
        """ Get probability estimation on x

            INPUT:
                alpha: float, alpha
                beta: float, beta
                l1: float, l1 penalty
                l2: float, l2 penalty
                nb_epoch: int, number of epochs to train
                batch_size: int,mini batch size
                early_stop_rounds:int, early_stop_rounds only applys when validation_set is used
                use_glorot: boolean, use glorot_uniform to initialize weights default is false
        """
        # parameters for training
        self.alpha = alpha
        self.beta = beta
        self.l1 = l1
        self.l2 = l2
        self.nb_epoch = nb_epoch
        self.iterations =0
        self.batch_size = batch_size
        self.eval_function= eval_function
        self.name=name
        self.drop_out =1-drop_out
        self.batch_normalization = batch_normalization
 
        
        # feature related parameters
        # model
        # n: squared sum of past gradients
        # z: weights
        # w: lazy weights
        # g: graident
        self.z = None
        self.n = None
        self.w = None
        self.g = 0.
        # initialize other params
        self.loss = 0.
        self.count = 0
        self.early_stop_rounds = early_stop_rounds
        self.use_glorot= use_glorot
        self.validation_set = validation_set
        self.validation_split = validation_split
    


    def _validate_y(self, y):
        y = column_or_1d(y, warn=True)
        self.classes_, y = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        return y

    def fit(self,X,y,verbose=1,shuffle=True,validation_set=None,validation_split = None,sample_weight=None):
        if validation_set==None:
            validation_set = self.validation_set
        if validation_split==None:
            validation_split = self.validation_split
        if validation_split!=None:
            X,y,xt,yt=test_train_data(X,y,p=validation_split)
            validation_set=[]
            validation_set.append(xt)
            validation_set.append(yt)
        y = self._validate_y(y)

        if self.batch_normalization:
            scaler = StandardScaler(with_mean=(not issparse(X)))

        classes_ = np.unique(y)
        nb_class = len(classes_)

        self.lb_y  = LabelBinarizer().fit(y)
        y = self.lb_y.transform(y)
        name = self.name
        eval_function = self.eval_function
        if self.z == None and self.n == None and self.w ==None:
            
            self.input_size = X.shape[1]
            self.n = np.zeros((self.input_size,nb_class))

            
            if self.use_glorot:
                self.w = self.glorot_uniform((self.input_size,nb_class))
                self.z = self.glorot_uniform((self.input_size,nb_class))
            else:
                self.w = np.zeros((self.input_size,nb_class))
                self.z = np.zeros((self.input_size,nb_class))

        batch_size = self.batch_size
        preds = []
        sample_size = X.shape[0]
        index_array = np.arange(sample_size)
        early_stop_rounds=self.early_stop_rounds
        best_auc = 0
        count=0
        for epoch in range(self.nb_epoch):
            if shuffle:
                np.random.shuffle(index_array)
            batches = make_batches(sample_size, batch_size)

            for batch_index, (batch_start, batch_end) in enumerate(batches):
                batch_ids = index_array[batch_start:batch_end]
                ins_batch = slice_X(X, batch_ids)
                if self.batch_normalization:
                    ins_batch = scaler.fit_transform(ins_batch)

                outs = self._predict(ins_batch)
                # if type(outs) != list:
                #     outs = [outs]
                target = y[batch_ids]
                self.update(ins_batch, outs, target)

            if early_stop_rounds and validation_set:
                tp = self.predict(validation_set[0])
                validation_auc = eval_function(validation_set[1],tp)
                if best_auc<validation_auc:
                    count=0
                    best_auc = validation_auc
                    f = open("best_ftrl_%s.mdl"%name,"wb")
                    cPickle.dump([self.z,self.n,self.w],f)
                    f.close()
                
                if count ==early_stop_rounds:
                    validation_auc = eval_function( validation_set[1],self.predict(validation_set[0]))
                    count=0
                    if best_auc>validation_auc:
                        break
                else:
                    count+=1


            if verbose>=1:
                self.loss = self.categorical_entropy(self._predict(X), y)
                if validation_set:
                    validation_loss = self.categorical_entropy(self._predict(validation_set[0]), self.lb_y.transform(validation_set[1]))
                    validation_auc = eval_function(validation_set[1],self.predict(validation_set[0]))
                    print('eoch %s\tcurrent_loss: %f\tvalid_loss: %f\tvalid_auc: %f'%(epoch,self.loss,validation_loss,validation_auc))
                else:
                    print('eoch %s\tcurrent loss: %f'%(epoch,self.loss))
        if early_stop_rounds and validation_set:
            f = open("best_ftrl_%s.mdl"%name,"rb")
            best_weights = cPickle.load(f)
            self.z = best_weights[0]
            self.n = best_weights[1]
            self.w = best_weights[2]
            f.close()
           


    def predict(self,X):
        preds = []
        preds = self._predict(X)
        return categorical_probas_to_classes(preds)

    def predict_proba(self,X):

        return self._predict_proba_lr(X)



    def _predict_proba_lr(self, X):
        """Probability estimation for OvR logistic regression.

        Positive class probabilities are computed as
        1. / (1. + np.exp(-self.decision_function(X)));
        multiclass is handled by normalizing that over all classes.
        """
        self.decision_function = self._predict
        prob = self.decision_function(X)
        prob *= -1
        np.exp(prob, prob)
        prob += 1
        np.reciprocal(prob, prob)
        if len(prob.shape) == 1:
            return np.vstack([1 - prob, prob]).T
        else:
            # OvR normalization, like LibLinear's predict_probability
            prob /= prob.sum(axis=1).reshape((prob.shape[0], -1))
            return prob



    def _predict(self, x):
        ''' Get probability estimation on x

            INPUT:
                x: features

            OUTPUT:
                probability of p(y = 1 | x; W)
        '''
        # parameters
        w = self.get_weights()
        # wTx is the inner product of w and x'

        # print w.shape
        wTx = safe_sparse_dot(x,w, dense_output=False)
        # print wTx.shape
        # print wTx
        # cache the current w for update stage
        self.w = w
        # bounded sigmoid function, this is the probability estimation
        p = self.sigmoid(wTx)
        # print p.shape

        return p


    def sigmoid(self,inX):  
        return 1.0 / (1 + np.exp(-inX))


    def bounded_sigmoid(self,inX):  
        return 1. / (1. + np.exp(-np.max(np.min(inX, 35.), -35.)))
    


    def update(self, x, p, y,sample_weight=None):
        ''' Update model using x, p, y

            INPUT:
                x: inputs
                p: click probability prediction of our model
                y: target

            MODIFIES:
                self.n: increase by squared gradient
                self.z: weights
        '''
        # gradient under logloss
        # use drop out
        mask = np.random.binomial(1, self.drop_out, size=y.shape)
        y = y*mask 

        p = np.array(p)
        # print p.shape
        diff= (p - y)
        # print diff.shape

        # print 'w.shape',self.w.shape
        self.g = safe_sparse_dot(x.T,diff)#.ravel()

        # print 'grad shape',self.g.shape
        sigma = (np.sqrt(self.n + self.g * self.g) - np.sqrt(self.n)) / self.alpha
        self.z += self.g - sigma * self.w
        self.n += self.g * self.g
        self.iterations +=1
        w = (np.sign(self.z) * self.l1 - self.z) / ((self.beta + np.sqrt(self.n)) / self.alpha + self.l2)
        idx_0 = np.abs(self.z) <= self.l1
        w[idx_0]=0
        self.w = w



def map_predict(map_X):
    x,batch_index,w = map_X
    ''' Get probability estimation on x

    INPUT:
        x: features

    OUTPUT:
        probability of p(y = 1 | x; W)
    '''


    wTx = safe_sparse_dot(x,w, dense_output=False)

    p = ParallelFtrlSoftmaxClassifier().sigmoid(wTx)
    return [p,batch_index]

def reduce_predict(reduce_preds):
    reduce_preds.sort(key=lambda x:x[1])
    reduced_res = []
    for res,batch_index in reduce_preds:
        reduced_res.append(res)
    reduced_res = np.concatenate(reduced_res)
    return reduced_res

from multiprocessing import Pool
class ParallelFtrlSoftmaxClassifier(BaseEstimator,WeightMixin):

    def get_fans(self,shape):
        fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
        fan_out = shape[1] if len(shape) == 2 else shape[0]
        return fan_in, fan_out
    def uniform(self,shape, scale=0.05):
        return np.random.uniform(low=-scale, high=scale, size=shape)

    def glorot_uniform(self,shape):
        fan_in, fan_out = self.get_fans(shape)
        s = np.sqrt(6. / (fan_in + fan_out))
        return self.uniform(shape, s)

    def binary_entropy(self,p, y):
        loss =-(np.array(y) * np.log(p) + (1.0 - np.array(y)) * np.log(1.0 - np.array(p)))
        return np.mean(loss)

    def categorical_entropy(self,p, y):
        mlogloss=0.0
        for n in range(y.shape[1]):
            loss_n =-(np.array(y)[:,n] * np.log(p)[:,n] + (1.0 - np.array(y)[:,n]) * np.log(1.0 - np.array(p)[:,n]))
            loss_n = np.mean(loss_n)
            mlogloss+=loss_n
        return mlogloss



    def get_weights(self):
        return self.w

    def get_z(self):
        return self.z


    def __init__(self,alpha=0.005, beta=1, l1=0.0, l2=0.0,nb_epoch=20,batch_size=128,early_stop_rounds=None,use_glorot=False,drop_out=0.0,eval_function=accuracy_score,validation_set=None,validation_split = None,name="default"):
        """ Get probability estimation on x

            INPUT:
                alpha: float, alpha
                beta: float, beta
                l1: float, l1 penalty
                l2: float, l2 penalty
                nb_epoch: int, number of epochs to train
                batch_size: int,mini batch size
                early_stop_rounds:int, early_stop_rounds only applys when validation_set is used
                use_glorot: boolean, use glorot_uniform to initialize weights default is false
        """
        # parameters for training
        self.alpha = alpha
        self.beta = beta
        self.l1 = l1
        self.l2 = l2
        self.nb_epoch = nb_epoch
        self.iterations =0
        self.batch_size = batch_size
        self.eval_function= eval_function
        self.name=name
        self.drop_out =1-drop_out
 

        # feature related parameters
        # model
        # n: squared sum of past gradients
        # z: weights
        # w: lazy weights
        # g: graident
        self.z = None
        self.n = None
        self.w = None
        self.g = 0.
        # initialize other params
        self.loss = 0.
        self.count = 0
        self.early_stop_rounds = early_stop_rounds
        self.use_glorot= use_glorot
        self.validation_set = validation_set
        self.validation_split = validation_split
    


    def _validate_y(self, y):
        y = column_or_1d(y, warn=True)
        self.classes_, y = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        return y


    def dispatch(self,X,n_jobs=8):
        index_array = np.arange(X.shape[0])
        sample_size = X.shape[0]
        batch_size = int(sample_size/n_jobs)
        batches = make_batches(sample_size, batch_size)
        pairs = []
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            batch_ids = index_array[batch_start:batch_end]
            X_batch = X[batch_ids]
            pairs.append([X_batch,batch_index,self.w])
        return pairs

    def _predict_parallel(self,X,n_jobs=8):
        pairs = self.dispatch(X,n_jobs=n_jobs)
        pool = Pool(n_jobs)
        map_X = pool.map(map_predict,pairs)
        reduced_res = reduce_predict(map_X)
        pool.close()
        pool.join()
        return reduced_res




    def _predict_thano(self,X):
        w = theano.shared(self.w)
        X = theano.shared(X)
        y_preds = T.nnet.sigmoid(theano.dot(X, w))
        y_preds = y_preds.eval()
        return y_preds



    def fit(self,X,y,verbose=1,shuffle=True,validation_set=None,validation_split = None,sample_weight=None):
        if validation_set==None:
            validation_set = self.validation_set
        if validation_split==None:
            validation_split = self.validation_split
        if validation_split!=None:
            X,y,xt,yt=test_train_data(X,y,p=validation_split)
            validation_set=[]
            validation_set.append(xt)
            validation_set.append(yt)
        y = self._validate_y(y)

        classes_ = np.unique(y)
        nb_class = len(classes_)

        self.lb_y  = LabelBinarizer().fit(y)
        y = self.lb_y.transform(y)
        name = self.name
        eval_function = self.eval_function
        if self.z == None and self.n == None and self.w ==None:
            
            self.input_size = X.shape[1]
            self.n = np.zeros((self.input_size,nb_class))
            if self.use_glorot:
                self.w = self.glorot_uniform((self.input_size,nb_class))
                self.z = self.glorot_uniform((self.input_size,nb_class))
            else:
                self.w = np.zeros((self.input_size,nb_class))
                self.z = np.zeros((self.input_size,nb_class))
                
        batch_size = self.batch_size
        preds = []
        sample_size = X.shape[0]
        index_array = np.arange(sample_size)
        early_stop_rounds=self.early_stop_rounds
        best_auc = 0
        count=0
        for epoch in range(self.nb_epoch):
            if shuffle:
                np.random.shuffle(index_array)
            batches = make_batches(sample_size, batch_size)
            for batch_index, (batch_start, batch_end) in enumerate(batches):
                batch_ids = index_array[batch_start:batch_end]
                ins_batch = slice_X(X, batch_ids)
                outs = self._predict_thano(ins_batch)
                # if type(outs) != list:
                #     outs = [outs]
                target = y[batch_ids]
                self.update(ins_batch, outs, target)
            if early_stop_rounds and validation_set:
                tp = self.predict(validation_set[0])
                validation_auc = eval_function(validation_set[1],tp)
                if best_auc<validation_auc:
                    count=0
                    best_auc = validation_auc
                    f = open("best_ftrl_%s.mdl"%name,"wb")
                    cPickle.dump([self.z,self.n,self.w],f)
                    f.close()
                
                if count ==early_stop_rounds:
                    validation_auc = eval_function( validation_set[1],self.predict(validation_set[0]))
                    count=0
                    if best_auc>validation_auc:
                        break
                else:
                    count+=1


            if verbose>=1:
                self.loss = self.categorical_entropy(self._predict(X), y)
                if validation_set:
                    validation_loss = self.categorical_entropy(self._predict(validation_set[0]), self.lb_y.transform(validation_set[1]))
                    validation_auc = eval_function(validation_set[1],self.predict(validation_set[0]))
                    print('eoch %s\tcurrent_loss: %f\tvalid_loss: %f\tvalid_auc: %f'%(epoch,self.loss,validation_loss,validation_auc))
                else:
                    print('eoch %s\tcurrent loss: %f'%(epoch,self.loss))
        if early_stop_rounds and validation_set:
            f = open("best_ftrl_%s.mdl"%name,"rb")
            best_weights = cPickle.load(f)
            self.z = best_weights[0]
            self.n = best_weights[1]
            self.w = best_weights[2]
            f.close()
           


    def predict(self,X):
        preds = []
        preds = self._predict(X)
        return categorical_probas_to_classes(preds)

    def predict_proba(self,X):

        return self._predict_proba_lr(X)



    def _predict_proba_lr(self, X):
        """Probability estimation for OvR logistic regression.

        Positive class probabilities are computed as
        1. / (1. + np.exp(-self.decision_function(X)));
        multiclass is handled by normalizing that over all classes.
        """
        self.decision_function = self._predict
        prob = self.decision_function(X)
        prob *= -1
        np.exp(prob, prob)
        prob += 1
        np.reciprocal(prob, prob)
        if len(prob.shape) == 1:
            return np.vstack([1 - prob, prob]).T
        else:
            # OvR normalization, like LibLinear's predict_probability
            prob /= prob.sum(axis=1).reshape((prob.shape[0], -1))
            return prob
        


    def _predict(self, x):
        ''' Get probability estimation on x

            INPUT:
                x: features

            OUTPUT:
                probability of p(y = 1 | x; W)
        '''
        # parameters
        w = self.get_weights()
        # wTx is the inner product of w and x'

        # print w.shape
        wTx = safe_sparse_dot(x,w, dense_output=False)
        # print wTx.shape
        # print wTx
        # cache the current w for update stage
        self.w = w
        # bounded sigmoid function, this is the probability estimation
        p = self.sigmoid(wTx)
        # print p.shape

        return p


    def sigmoid(self,inX):  
        return 1.0 / (1 + np.exp(-inX))


    def bounded_sigmoid(self,inX):  
        return 1. / (1. + np.exp(-np.max(np.min(inX, 35.), -35.)))
    
    def update(self, x, p, y,sample_weight=None):
        ''' Update model using x, p, y

            INPUT:
                x: inputs
                p: click probability prediction of our model
                y: target

            MODIFIES:
                self.n: increase by squared gradient
                self.z: weights
        '''
        # gradient under logloss
        # use drop out
        mask = np.random.binomial(1, self.drop_out, size=y.shape)
        y = y*mask 

        p = np.array(p)
        # print p.shape
        diff= (p - y)
        # print diff.shape

        # print 'w.shape',self.w.shape
        self.g = safe_sparse_dot(x.T,diff)#.ravel()

        # print 'grad shape',self.g.shape
        sigma = (np.sqrt(self.n + self.g * self.g) - np.sqrt(self.n)) / self.alpha
        
        self.z += self.g - sigma * self.w
        self.n += self.g * self.g
        self.iterations +=1
        w = (np.sign(self.z) * self.l1 - self.z) / ((self.beta + np.sqrt(self.n)) / self.alpha + self.l2)
        idx_0 = np.abs(self.z) <= self.l1
        w[idx_0]=0
        self.w = w



class ElasitcNetRegression(BaseEstimator):

    def get_fans(self,shape):
        fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
        fan_out = shape[1] if len(shape) == 2 else shape[0]
        return fan_in, fan_out
    def uniform(self,shape, scale=0.05):
        return np.random.uniform(low=-scale, high=scale, size=shape).astype(theano.config.floatX)

    def glorot_uniform(self,shape):
        fan_in, fan_out = self.get_fans(shape)
        s = np.sqrt(6. / (fan_in + fan_out))
        return self.uniform(shape, s).astype(theano.config.floatX)

    def __init__(self,lr=0.005,l1=0.0, l2=0.0,nb_epoch=20,batch_size=128,early_stop_rounds=0,validation_set=None,validation_split=None,use_glorot=True,name='elastic_net.mdl'):
        """ Get probability estimation on x

            INPUT:
                alpha: float, alpha
                beta: float, beta
                l2: float, l2 penalty
                nb_epoch: int, number of epochs to train
                batch_size: int,mini batch size
                early_stop_rounds:int, early_stop_rounds only applys when validation_set is used
                use_glorot: boolean, use glorot_uniform to initialize weights default is false
        """
        # parameters for training
        self.lr = lr
        self.l1 = l1
        self.l2 = l2
        self.nb_epoch = nb_epoch
        self.iterations =0
        self.batch_size = batch_size
        self.early_stop_rounds = early_stop_rounds
        self.use_glorot= use_glorot
        self.validation_set = validation_set
        self.validation_split = validation_split
        self.name = name

    def _validate_y(self, y):
        y = column_or_1d(y, warn=True)
        self.classes_, y = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        return y

    def fit(self,X,y,verbose=1,shuffle=True):



        '''
            create shared X,y
        '''


        '''
            if use validation set to perform early_stop_rounds
        '''
        validation_set = self.validation_set
        if validation_set==None:
            validation_set = self.validation_set
            validation_split = self.validation_split
        if validation_split!=None:
            X,y,X_test,y_test=test_train_data(X,y,p=validation_split)
            validation_set=[]
            validation_set.append(X_test)
            validation_set.append(y_test)
        
        '''
            get defalut parameters
        '''
        lr = self.lr
        batch_size = self.batch_size
        sample_size = X.shape[0]
        index_array = np.arange(sample_size)
        y = self._validate_y(y)
        self.X = X
        self.y = y
        if shuffle:
            np.random.shuffle(index_array)

        '''
            build shared inputs and labels
        '''
        shared_X = theano.shared(np.asarray(X,
                                               dtype=theano.config.floatX),
                                 borrow=True)
        shared_y = theano.shared(np.asarray(y,
                                               dtype=theano.config.floatX),
                                 borrow=True)
        shared_y = T.cast(shared_y, 'int32')
        self.shared_X = shared_X
        self.shared_y = shared_y
        n_in = X.shape[1]
        n_out = self.n_classes_
        




        '''
            define symbolic parameters
        '''
        # index to a [mini]batch
        self.batch_index = T.ivector()
        self.x_theano = T.matrix('x')  # data
        self.y_theano = T.ivector('y')  # labels, presented as 1D vector of [int] labels
        # initial weights if using glorot uniform


        if self.use_glorot:
            w = self.glorot_uniform([n_in,n_out])
        else:
            w = np.zeros([n_in, n_out],dtype=theano.config.floatX)

        self.W = theano.shared(
                value=w,
                name='W',
                borrow=True,
                )
        # predict_proba symbolic
        self.p_y_given_x = self._predict_proba()
        # cost symbolic
        self.cost = self._logloss()
        # gradient symbolic
        self.g = self._gradient()
        # gradient symbolic
        self.updates = self._update(lr)
        train_on_batch = theano.function(
            inputs=[self.batch_index],
            outputs=self.cost,
            updates=self.updates,
            givens={
                self.x_theano: shared_X[self.batch_index],
                self.y_theano: shared_y[self.batch_index]
            }
        )

        print 'training the model'
        nb_epoch = self.nb_epoch
        best_loss = np.inf
        best_epoch = 0
        batches = make_batches(sample_size, batch_size)
        for epoch in range(nb_epoch):
            loss = 0
            for batch_idx, (batch_start, batch_end) in enumerate(batches):
                batch_ids = index_array[batch_start:batch_end].astype(np.int32)
                loss += train_on_batch(batch_ids)
            loss /=len(batches)
            if validation_set!=None and self.early_stop_rounds>0:
                X_test,y_test = validation_set
                y_test_preds = self.predict_proba(X_test)
                valid_loss = self.evaluate(y_test,y_test_preds)
                if best_loss>=valid_loss:
                    best_epoch = epoch
                    best_loss = valid_loss
                    self.wait=0
                    self.save_weights(self.name)
                else:
                    if self.wait >= self.early_stop_rounds:
                        self.load_weights(self.name)
                        if verbose>0:
                            print "best epoch: {best_epoch}, best loss: {best_loss}".format(best_epoch=best_epoch,best_loss=best_loss)
                        break

                if verbose>0:
                    print "epoch {epoch}: train loss: {loss} valid loss: {valid_loss}".format(epoch=epoch,loss=loss,valid_loss=valid_loss)
            else:
                if verbose>0:
                    print "epoch {epoch}: train loss: {loss}".format(epoch=epoch,loss=loss)
        self.coef_ = self.W.get_value(borrow=True)

    def save_weights(self,file_path = ""):
        assert file_path!=""
        w = self.W.get_value(borrow=True)
        np.save(file_path,w)

    def load_weights(self,file_path = ""):
        w = self.W.get_value(borrow=True)
        self.W = theano.shared(
                value=w,
                name='W',
                borrow=True,
                )

    def evaluate(self,y,y_preds):
        eval_loss = self._logloss()
        eval_function = theano.function(
                inputs = [self.p_y_given_x,self.y_theano],
                outputs = eval_loss,
                allow_input_downcast=True,
            )
        return eval_function(y_preds,y)

    # def categorical_crossentropy(self):
    #     _EPSILON = 1e-8
    #     self.p_y_given_x /= self.p_y_given_x.sum(axis=-1, keepdims=True)+self.l1*T.abs_(T.sum(self.W))+self.l2*T.dot(self.W.ravel(),self.W.ravel())
    #     # avoid numerical instability with _EPSILON clipping
    #     self.p_y_given_x = T.clip(self.p_y_given_x, _EPSILON, 1.0 - _EPSILON)
    #     return T.nnet.categorical_crossentropy(self.p_y_given_x, self.y_theano)

    def predict_proba(self,X):
        X = X.astype(np.float32)
        predict_proba_function = theano.function(
            inputs=[self.x_theano],
            outputs=self._predict_proba(),
        )
        return predict_proba_function(X)

    def predict(self,X):
        X = X.astype(np.float32)
        predict_function = theano.function(
            inputs=[self.x_theano],
            outputs=self._predict(),
        )
        return predict_function(X)
    
    # def _logloss(self):
    #     return -T.mean(T.log(self.p_y_given_x)[T.arange(self.y_theano.shape[0]), self.y_theano])

    def _logloss(self):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(self.y_theano.shape[0]), self.y_theano]+self.l1*T.abs_(T.sum(self.W))+self.l2*T.dot(self.W.ravel(),self.W.ravel()))

    def _predict(self):
        y_pred_theano = T.argmax(self._predict_proba(), axis=1)
        return y_pred_theano

    def _predict_proba(self):
        y_pred_theano = T.nnet.softmax(T.dot(self.x_theano, self.W))
        return y_pred_theano


    def _gradient(self):
        g = T.grad(cost=self._logloss(), wrt=self.W)
        return g

    def _update(self,lr):
        updates = [(self.W, self.W - lr * self.g)]
        return updates
    
