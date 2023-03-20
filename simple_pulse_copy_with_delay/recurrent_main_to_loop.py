##########################################################
#                 Author C. Jarne                        #
#            recurrent_main_to_loop.py  (ver 1.0)        #                       
#          Perceptual Decision Making Task               #
#             (low edge triggered)                       #                
#                                                        #
# MIT LICENCE                                            #
##########################################################

#Python Basics
import numpy as np
import matplotlib.pyplot as plt
import time

#Keras and TF basics
import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers.core import Dense
from keras.callbacks import ModelCheckpoint, Callback   #, warnings
#from keras.layers.recurrent import SimpleRNN
from keras.layers import Dense,  SimpleRNN, Activation, Dropout #TimeDistributed,
from keras.utils import plot_model
from keras import metrics
from keras import optimizers
from keras import regularizers
from keras.layers import Input
from keras.constraints import Constraint
 


# taking dataset from function
#from generate_perceptual_dm import *
from generate_data_set_time_pulse import *

from net_constraint_create import *

#Early stop
class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='val_loss', value=0.000001, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value   = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print(" Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True

#Fun to call fron loop_to_call.py

def DM_fun(t,N_rec,base,base_plot):
    lista_distancia=[]
    #Parameters
    sample_size      =4*15050
    epochs           = 20
    mem_gap          = t

    x_train,y_train, mask,seq_dur = generate_trials(sample_size,mem_gap) #time loop
    
    #Network model construction
    seed(None)# change seed    
    model = Sequential()
    model.add(SimpleRNN(units=N_rec,return_sequences=True, input_shape=(None, 1), kernel_constraint=NonNegLast_input(), kernel_initializer='glorot_uniform',recurrent_initializer= pepe,recurrent_constraint=None, use_bias=True))           #,use_bias=True,bias_initializer='zeros')) #defaults for the recurrent model!
    model.add(Dense(units=1,input_dim=N_rec))
    
    model.save(base+'/'+base_plot[-4]+base_plot[-3]+'_00_initial.hdf5')
    
    # Model Compiling:
    ADAM           = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999,epsilon=1e-08, decay=0.0001, clipnorm=1.0)
    model.compile(loss = 'mse', optimizer=ADAM, sample_weight_mode="temporal")

    # Saving weigths
    filepath       = base+'/perceptual_DM_weights-{epoch:02d}.hdf5'
    #checkpoint    = ModelCheckpoint(filepath, monitor='accuracy')
    #checkpoint     = ModelCheckpoint(filepath)
    callbacks      = [EarlyStoppingByLossVal(monitor='loss', value=0.00005, verbose=1), ModelCheckpoint(filepath, monitor='val_loss', save_best_only=False, verbose=1),]


    #from keras.callbacks import TensorBoard
    #tensorboard = TensorBoard(log_di r='./logs', histogram_freq=0, write_graph=True, write_images=False)

    #Training:
    history      = model.fit(x_train[50:sample_size,:,:], y_train[50:sample_size,:,:], epochs=epochs, batch_size=64, callbacks = callbacks,     sample_weight=mask[50:sample_size,:])

    #callbacks=[tensorboard]
    #callbacks = [checkpoint]

    # Model Testing: 
    x_pred = x_train[0:50,:,:]
    y_pred = model.predict(x_pred)

    print("x_train shape:\n",x_train.shape)
    print("x_pred shape\n",x_pred.shape)
    print("y_train shape\n",y_train.shape)

    fig     = plt.figure(figsize=(6,8))
    fig.suptitle("\"Perceptual DM\" Data Set Trainined Output \n (amplitude in arb. units time in mS)",fontsize = 20)
    for ii in np.arange(10):
        plt.subplot(5, 2, ii + 1)           
        plt.plot(x_train[ii, :, 0],color='g',label="Input A")
        plt.plot(y_train[ii, :, 0],color='gray',linewidth=2,label="Expected Output")
        plt.plot(y_pred[ii, :, 0], color='r',label="Predicted Output")
        plt.ylim([-2.5, 2.5])
        plt.legend(fontsize= 5,loc=3)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        a=y_train[ii, :, 0]
        b=y_pred[ii, :, 0]
        a_min_b = np.linalg.norm(a-b)      
        lista_distancia.append(a_min_b)    
    figname =  base_plot+"/data_set_sample_trained.png"       
    plt.savefig(figname,dpi=200)
    plt.close()
    #plt.show()
    


    print ("history keys",(history.history.keys()))

    #print("--- %s to train the network seconds ---" % (time.time() - start_time))

    fig     = plt.figure(figsize=(8,6))
    plt.grid(True)
    plt.plot(history.history['loss'])
    plt.title('Model loss during training')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    figname = base_plot+"/model_loss_"+str(N_rec)+".png" 
    plt.savefig(figname,dpi=200)

    print(model.summary())
    #plot_model(model, to_file=base_plot+'/model.png')



   


