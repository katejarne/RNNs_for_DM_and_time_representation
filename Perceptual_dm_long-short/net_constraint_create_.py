###################################################
#   Constraints for initialization and training   #
#   To be called from recurrent_main_to_loop.py   #
#                    V0.1                         #
###################################################

import numpy as np
import keras
import keras.backend as K
from keras.constraints import Constraint
from keras.initializers import Initializer
import tensorflow as tf

#Dale's Law
#constraints over columns represents all output conection for each units.

###  For initialization
g=1
#N_rec=t

def constrain(N_rec):

    # Two gaussian populations
    def my_init_exi_ini(shape, dtype=None):
        mu_ex=0.05
        mu_in=-0.05
        sigma=np.sqrt(1/(N_rec))
        exi= g*np.random.normal(mu_ex, sigma, (int(N_rec),int(N_rec/2)))
        ini= g*np.random.normal(mu_in, sigma, (int(N_rec),int(N_rec/2)))        
        #exi= g*np.random.normal(mu_ex, sigma, (int(N_rec),int(3*N_rec/4)))
        #ini= g*np.random.normal(mu_in, sigma, (int(N_rec),int(N_rec/4)))
        #shape      = np.concatenate((ini,exi), axis=1)
        shape      = np.concatenate((exi,ini), axis=1)
        return K.variable( shape, dtype=dtype )


    def my_init_rec(shape, name=None,dtype=tf.float32):
        shape      = 1*np.identity(N_rec)
        return K.variable(shape, name=name, dtype=dtype)


    pepe= keras.initializers.RandomNormal(mean=0.0, stddev=np.sqrt(float(1)/float((N_rec))), seed=None)

    ####   For training  

    #(RNN)  a fraction of units are excitatory another inhibitory
    class NonNegLast(Constraint):
        def __call__(self, w):
            first_cols= w[:,0:int(N_rec/4)]*K.cast(K.less_equal(w[:,0:int(N_rec/4)], 0.0), K.floatx())
            last_cols= w[:,int(N_rec/4):int(N_rec)]*K.cast(K.greater_equal(w[:,int(N_rec/4):int(N_rec)], 0.0), K.floatx())                                
            full_matrix = K.concatenate([first_cols,last_cols],1)
            return full_matrix

    #input units excitatory  
    class NonNegLast_input(Constraint):
        def __call__(self, w):  
            last_rowss_01= w[:, 0:N_rec]*K.cast(K.greater_equal(w[:,0:N_rec], 0.), K.floatx())             
            full_w_      = last_rowss_01
            return full_w_    






