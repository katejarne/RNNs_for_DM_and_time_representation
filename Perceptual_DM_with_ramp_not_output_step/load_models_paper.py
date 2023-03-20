import os
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pylab import grid
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from pylab import grid
from scipy.stats import norm
from numpy import linalg as LA
import matplotlib.mlab as mlab
from keras.utils import CustomObjectScope

from keras.models import Sequential,load_model
from keras.layers.core import Dense
from keras.callbacks import ModelCheckpoint
#from keras.layers.recurrent import SimpleRNN
from keras.layers import TimeDistributed, Dense, Activation, Dropout,SimpleRNN
from keras.utils import plot_model
from keras import metrics
from keras import optimizers
from keras import regularizers
from keras import initializers
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from mpl_toolkits.mplot3d import Axes3D

# Para coustomizar el constraint!!!!
from keras.constraints import Constraint

import tensorflow as tf

# taking dataset from function:

#from generate_perceptual_dm import *
from generate_perceptual_dm_ramp import *  
#from generate_perceptual_dm_ramp_1 import *


from net_constraint_create import *

# To print network status
from print_status_2_inputs_paper import *

#Parameters:
sample_size_3       =4#10#4
mem_gap             = 20
sample_size         = 200 # Data set to print some results
lista_distancia_all =[]
lista_freq_sample_net=[]
# Generate a data Set to study the Network properties:


con_matrix_list_pos            = []
con_matrix_list_neg            = []
con_matrix_list                = []
full_eigen_list=[]
j2_full_eigen_list=[]
H_number_list=[]

net_freq=[]
def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


#r_dir="/home/kathy/Desktop/Neuronal_networks/2022-kreso-proyect/perceptual_decision_making/weights/2023-01-16/"
r_dir="/home/kathy/Desktop/2022-kreso-proyect/percetpual_DM_ramp/weights/perceptual_DM_ramp_simp/ramp_dur_5/"


plot_dir="plots"


lista_neg     =[]
lista_pos     =[]
total         =[]
lista_neg_porc=[]
lista_pos_porc=[]
lista_tot_porc=[]

string_name_list=[]

N_rec=50#400#50
g=1
import fnmatch
import re
for root, sub, files in os.walk(r_dir):
    files = sorted(files)
    
    for i,f in enumerate(files):
        print(f)
        if  fnmatch.fnmatch(f, '*20.hdf5'):  # or fnmatch.fnmatch(f, '*initial.hdf5') :#
           print("file: ",f)
           r_dir=root
           mem_gap_= os.path.basename(root) #(-4:)
           mem_gap_= r_dir[-45:-15]
           print(root)
           match = re.search(r'\d+', mem_gap_)
           if match:
              mem_gap  = int(match.group())
              print("Number:", mem_gap)
           else:
              print("No se encontró ningún número en el nombre de archivo")
                    
           print("mem_gap",mem_gap_)
           #string_name=root[-10:-7]
           string_name=root[-19:]
           #string_name=root[-20:-12]
           #string_name=root[-1:]
           amplitude=mem_gap
           
           #x_train,y_train, mask,seq_dur  = generate_trials(sample_size,mem_gap) # if use generate_perceptual_dm.py
           x_train,y_train, mask,seq_dur  = generate_trials(sample_size, amplitude) #if use generate_perceptual_dm_ramp.py
             
           test                           = x_train[0:1,:,:] # Here you select from the generated data set which is used for test status
           test_set                       = x_train[0:20,:,:]
           y_test_set                     = y_train[0:20,:,0]
           
           print("string_name",string_name)
           string_name_list.append(string_name)
           print("r_dir",r_dir)

           #General network model construction:

           model = Sequential()

           #model = load_model(r_dir+"/"+f)      
           #seed(2)

           #model.reset_states()
           #model = load_model(r_dir+"/"+f, custom_objects={'NonNegLast':NonNegLast})
           model = load_model(r_dir+"/"+f,  custom_objects={'NonNegLast':NonNegLast, 'NonNegLast_input':NonNegLast_input, 'my_init_exi_ini' : my_init_exi_ini,'my_init_rec':my_init_rec},compile = False)
           # Compiling model for each file:
           model.compile(loss = 'mse', optimizer='Adam', sample_weight_mode="temporal")

           print("-------------",i)
           #Esto me permite imprimir los pesos de la red y la estructura!!!
           for jj, layer in enumerate(model.layers):
               print("i-esima capa: ",jj)
               print(layer.get_config(), layer.get_weights())

           pesos     = model.layers[0].get_weights()
           pesos__   = model.layers[0].get_weights()[0]
           pesos_in  = pesos[0]
           pesos_out = model.layers[1].get_weights()
           pesos     = model.layers[0].get_weights()[1] 
           biases   = model.layers[0].get_weights()[2]
           '''

           pesos    = model.layers[1].get_weights()
           pesos__  = model.layers[1].get_weights()[0]
           pesos_in = pesos[1]
           pesos    = model.layers[1].get_weights()[1] 
           #biases   = model.layers[1].get_weights()[2]
           #pepe_4 = model.layers[0].get_weights()[3] 
           '''

           N_rec                          =len(pesos_in[0])  # it has to match the value of the recorded trained network
           neurons                        = N_rec
           colors                         = cm.rainbow(np.linspace(0, 1, neurons+1))


           print( "h",model.layers[0].states[0])

           print("-------------\n-------------")   
           print("pesos:\n:",pesos)
           print("-------------\n-------------")
           print("N_REC:",N_rec)
           unidades        = np.arange(len(pesos))
           conection       = pesos

           con_matrix_list.append(pesos)
           print("array: ",np.arange(len(pesos)))       
           #print("biases: ",biases)

           print("##########################\n ##########################")
           print("conection",conection)       
           print("##########################\n ##########################")

           histo_lista    =[]
           array_red_list =[]

           peso_mask  = 0.001 # 0.1# 0.011
           peso_mask_2=-0.001#-0.1#-0.011

           # Test anulando las conecciones que son mas debiles que el x% del max o el minimo

           conection_usar =conection#
           conection_sym  =0.5*(conection+tf.transpose(conection))
           #conection_usar[(conection_usar < peso_mask) & (conection_usar > peso_mask_2)] = 0
           #np.fill_diagonal(conection_usar, 0.0)
           #model.layers[0].set_weights([pesos_in,conection_usar])
           model.layers[0].set_weights([1*pesos_in,1*conection_usar, 1*biases])
           w, v = LA.eig(conection_usar)

           print("Autovalores:\n", w)
           print("Autovectores:\n",v)
           print("Distancia:", np.sqrt(w.real*w.real+w.imag*w.imag))

           lista_dist  = np.c_[w,w.real]
           lista_dist_2= np.c_[w,abs(w.real)]
           maximo      = max(lista_dist, key=lambda item: item[1])

           maximo_2= max(lista_dist_2, key=lambda item: item[1])
           marcar  = maximo[0]
           marcar_2= maximo_2[0]

           print("Primer elemento",maximo)
           print("Maximo marcar",marcar)


           frecuency=0
           if marcar_2.imag==0:
               frecuency =0
           else: 
               frecuency =abs(float(marcar_2.imag)/(3.14159*float(marcar_2.real)))

           print( "frecuency",frecuency)


           lista_modulos_=np.sqrt(w.real*w.real+w.imag*w.imag)
           lista_freq_=1000*np.absolute(w.imag/(3.14159*w.real))
           #lista_img_= np.absolute(float(w.imag))
           ##

           w_2=list(w)

           list_dist_ordered=sorted(w_2, key=lambda x: abs(x.imag) )
           print("list_dist sorted", list_dist_ordered)

           j2 = [i for i in w_2 if abs(i.real*i.real+i.imag*i.imag) > 1 and i.imag!=0]
           #j2_full_eigen_list.append(j2)
           print (j2)

           if len(j2)>0:
               ultimo= max(j2,key= np.abs)# np.imag)
           else:
               ultimo =marcar_2
          # else:
          #    j2 = [i for i in w_2 if abs(i.real*i.real+i.imag*i.imag) >1]
          #    ultimo= max(j2,key= np.abs)

           print("modulos",lista_modulos_)

           print("j2 ",j2 )
           print("j2 ultimo",ultimo )
           #time.sleep(5)
           
           frecuency_ultimo =1000*abs(float(ultimo.imag)/(2*3.14159*float(ultimo.real)))
           net_freq.append([string_name,frecuency_ultimo])
           frecuency_ultimo_="%.2f" % frecuency_ultimo
           #Symetric part
           lista_modulos_cuad=  [i**2 for i in lista_modulos_]

           #Henrici’s departure from normality
           H_number=np.sqrt(np.power(np.linalg.norm(conection_usar),2)-sum(lista_modulos_cuad))/np.linalg.norm(conection_usar)
           H_number_list.append([mem_gap,H_number])

           w_s, v_s = LA.eig(conection_sym)


           lista_dist_s  = np.c_[w_s,w_s.real]
           lista_dist_2_s= np.c_[w_s,abs(w_s.real)]
           maximo_s      = max(lista_dist_s, key=lambda item: item[1])

           maximo_2_s= max(lista_dist_2_s, key=lambda item: item[1])
           marcar_s  = maximo_s[0]
           marcar_2_s= maximo_2_s[0]

           ################ Fig Eigenvalues ########################

           plt.figure(figsize=cm2inch(8.5,7.5))
           plt.scatter(w.real,w.imag,color="hotpink",label="Eigenvalue Spectrum",s=2)#Total of: "+str(len(w.real))+" values")
           plt.scatter(ultimo.real,ultimo.imag,color="blue",label="Eigenvalue comp"+str(ultimo)+ "Freq: "+str(frecuency_ultimo_),s=5)
           #plt.scatter(w[-3].real,w[-3].imag,color="blue",label="Eigenvalue spectrum\n"+str(marcar_2),s=5)
           #plt.scatter(w[-4].real,w[-4].imag,color="blue",label="Eigenvalue spectrum\n"+str(marcar_2),s=5)
           #plt.scatter(w.real,w.imag,color="plum",label="Eigenvalue spectrum\n Total of: "+str(len(w.real))+" values")
           # for plotting circle line:
           a = np.linspace(0, 2*np.pi, 500)
           cx,cy = np.cos(a), np.sin(a)
           plt.plot(cx, cy,'--', alpha=.5, color="dimgrey") # draw unit circle line
           #plt.plot(2*cx, 2*cy,'--', alpha=.5)
           plt.scatter(marcar.real,marcar.imag,color="red", label="Eigenvalue max real part \n" +str(marcar_2),s=5)
           #plt.scatter(marcar_2.real,marcar_2.imag,color="blue", label="Eigenvalue with maximum abs(Real part)\n"+"Frecuency: "+str(frecuency))
           plt.plot([0,marcar.real],[0,marcar.imag],'-',color="grey")
           #plt.plot([0,marcar_2.real],[0,marcar_2.imag],'k-')
           plt.axvline(x=1,color="salmon",linestyle='--')
           plt.xticks(fontsize=4)
           plt.yticks(fontsize=4)
           plt.ylim([-1.4, 1.75])
           plt.xlim([-1.5, 1.65])
           plt.xlabel(r'$Re( \lambda)$',fontsize = 11)
           plt.ylabel(r'$Im( \lambda)$',fontsize = 11)
           #plt.legend(fontsize= 8,loc=1)            
           leg = plt.legend(fontsize= 5,loc=1)
           leg.get_frame().set_linewidth(0.0)
           #leg = plt.legend()
           #leg.get_frame().set_linewidth(0.0)    
           plt.savefig(plot_dir+"/autoval_"+str(i)+"_"+str(mem_gap)+"_"+str(peso_mask)+"_"+str(string_name)+".png",dpi=300, bbox_inches = 'tight')
           plt.close()
  
           ####################à

           
           plt.figure(figsize=cm2inch(8.5,7.5))
           plt.scatter(w_s.real,w_s.imag,color="hotpink",label="Eigenvalue Sym spectrum\n"+str(marcar_2_s),s=2)#Total of: "+str(len(w.real))+" values")
           #plt.scatter(w.real,w.imag,color="plum",label="Eigenvalue spectrum\n Total of: "+str(len(w.real))+" values")
           # for plotting circle line:
           a = np.linspace(0, 2*np.pi, 500)
           cx,cy = np.cos(a), np.sin(a)
           plt.plot(cx, cy,'--', alpha=.5, color="dimgrey") # draw unit circle line
           #plt.plot(2*cx, 2*cy,'--', alpha=.5)
           plt.scatter(marcar_s.real,marcar_s.imag,color="red", label="Eigenvalue maximum real part",s=5)
           #plt.scatter(marcar_2.real,marcar_2.imag,color="blue", label="Eigenvalue with maximum abs(Real part)\n"+"Frecuency: "+str(frecuency))
           plt.plot([0,marcar_s.real],[0,marcar_s.imag],'-',color="grey")
           #plt.plot([0,marcar_2.seal],[0,marcar_2.imag],'k-')
           plt.axvline(x=1,color="salmon",linestyle='--')
           plt.xticks(fontsize=4)
           plt.yticks(fontsize=4)
           plt.ylim([-1.4, 1.75])
           plt.xlim([-1.5, 1.65])
           plt.xlabel(r'$Re( \lambda)$',fontsize = 11)
           plt.ylabel(r'$Im( \lambda)$',fontsize = 11)
           #plt.legend(fontsize= 8,loc=1)            
           leg = plt.legend(fontsize= 7,loc=1)
           leg.get_frame().set_linewidth(0.0)
           #leg = plt.legend()
           #leg.get_frame().set_linewidth(0.0)    
           #plt.savefig(plot_dir+"/autoval_sym_"+str(i)+"_"+str(mem_gap)+"_"+str(peso_mask)+"_"+str(string_name)+".png",dpi=300, bbox_inches = 'tight')
           plt.close()
         


           # Model Testing: 
           x_pred = x_train[0:10,:,:]
           y_pred = model.predict(x_pred)
           ###################### for series ###########################

           fig     = plt.figure(figsize=cm2inch(19,7))
           plt.subplot(3, 2, 2) 
           frameon=False
           left, width = .25, .5
           bottom, height = .25, .5
           right = left + width
           top = bottom + height
           plt.text(0.5*(left+right), 0.5*(bottom+top), 'Instance '+str(i),
           horizontalalignment='center',
           verticalalignment='center',
           fontsize=20, color='red')
           plt.yticks([])
           plt.xticks([])
           plt.subplot(3, 2, 4) 
           frameon=True
           #plt.plot(x_train[1,:,0]+0.1* np.random.randn(seq_dur),color='g',label='Input Set')
           #plt.plot(x_train[1,:,1]+0.1* np.random.randn(seq_dur),color='pink',label='Input Reset')
           plt.plot(x_train[1,:,0],color='g',label='Input Reset')
           #plt.plot(x_train[1,:,1],color='pink',label='Input Set')

           plt.plot(y_train[1,:, 0],color='grey',linewidth=3,label='Target Output')  
           plt.plot(y_pred[1,:, 0], color='r',linewidth=2,label=' Output')
           #plt.xlim(0,seq_dur+1)
           plt.ylim([-0.5, 1.5])
           plt.yticks([])
           plt.xlabel('time [mS]',fontsize = 10)
           #plt.xticks(np.arange(0,seq_dur+1,50),fontsize = 8)
           plt.xticks(np.arange(0,250+1,50),fontsize = 8)
           plt.legend(fontsize= 3,loc=1)

           plt.subplot(1, 2, 1) 
           plt.scatter(w.real,w.imag,color="hotpink",label="Eigenvalue spectrum\n ",s=2)#Total of: "+str(len(w.real))+" values")
           #plt.scatter(w.real,w.imag,color="plum",label="Eigenvalue spectrum\n Total of: "+str(len(w.real))+" values")
           # for plotting circle line:
           a = np.linspace(0, 2*np.pi, 500)
           cx,cy = np.cos(a), np.sin(a)
           plt.plot(cx, cy,'--', alpha=.5, color="dimgrey") # draw unit circle line
           #plt.plot(2*cx, 2*cy,'--', alpha=.5)
           plt.scatter(marcar.real,marcar.imag,color="red", label="Eigenvalue maximum real part",s=5)
           #plt.scatter(marcar_2.real,marcar_2.imag,color="blue", label="Eigenvalue with maximum abs(Real part)\n"+"Frecuency: "+str(frecuency))
           plt.plot([0,marcar.real],[0,marcar.imag],'-',color="grey")
           #plt.plot([0,marcar_2.real],[0,marcar_2.imag],'k-')
           plt.axvline(x=1,color="salmon",linestyle='--')
           plt.xticks(fontsize=4)
           plt.yticks(fontsize=4)
           plt.ylim([-1.4, 1.75])
           plt.xlim([-1.5, 1.65])
           plt.xlabel(r'$Re( \lambda)$',fontsize = 11)
           plt.ylabel(r'$Im( \lambda)$',fontsize = 11)
           #plt.legend(fontsize= 8,loc=1)            
           leg = plt.legend(fontsize= 4,loc=1)
           leg.get_frame().set_linewidth(0.0)

           plt.savefig(plot_dir+"/sec_autoval_"+str(i)+"_"+str(mem_gap)+"_"+str(peso_mask)+"_"+str(string_name)+"_.png",dpi=300, bbox_inches = 'tight')
           plt.close()
           ################### 3d plot

           xs = np.linspace(0, 51, 100)
           zs = np.linspace(-1.2, 1.2, 100)
           X, Z = np.meshgrid(xs, zs)
           Y = 1


           ys = np.linspace(0, 1.2, 100)
           X_, Y_ = np.meshgrid(xs, ys)

           Z_=0   
           
          
          

           for ii in unidades:
               histo_lista.extend(pesos[ii])

           media= np.average(histo_lista)


           # best fit of data
           (mu, sigma) = norm.fit(histo_lista)
           # the histogram of the data
           n, bins, patches = plt.hist(histo_lista, 200, facecolor='green', alpha=0.75)

           # add a 'best fit' line
           #y = mlab.normpdf( bins, mu, sigma)
           y =  norm.pdf( bins, mu, sigma)
           #plt.figure(figsize=(14,10)) 
           

           print("x_train shape:\n",x_train.shape)
           print("x_pred shape\n",x_pred.shape)
           print( "y_train shape\n",y_train.shape)

           lista_distancia=[]
           
           
                   
           #################Conectivity matrix: positive or excitatory weights ##########
           #0.1
           fig2= plt.figure(figsize=(15,5)) 
           
           conection_pos  = np.ma.masked_where(abs(conection) < peso_mask, conection)
           import matplotlib.colors as clr
           from matplotlib.colors import BoundaryNorm
           ax=[]
           
           cbar_max  = 1#0.75
           cbar_min  =-1#-0.75
           cbar_step =0.025 #0.012#0.025
           out=pesos_out[0]
           #out_rev= out[::-1]
          
             
           # define the colormap
           cmap = plt.get_cmap('Spectral')
           #cmap = plt.get_cmap('viridis')
           # extract all colors from the .jet map
           cmaplist = [cmap(i) for i in range(cmap.N)]
           # create the new map
           cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

           # define the bins and normalize and forcing 0 to be part of the colorbar!
           bounds = np.arange(np.min(conection_pos),np.max(conection_pos),.05)
           idx=np.searchsorted(bounds,0)
           bounds=np.insert(bounds,idx,0)
           norm_ = BoundaryNorm(bounds, cmap.N)  
           
        
           #cmap = clr.LinearSegmentedColormap.from_list('custom blue', ['#244162','#DCE6F1'], N=256)
         
           cmap.set_bad(color='white')
           
           
           
           ## W in
           ax.append(fig2.add_subplot(1,3,1))
           plt.title(r'$W^{in}$', fontsize = 20)
           im1=plt.imshow(pesos_in.T, cmap=cmap,interpolation="none",label=r'$W^{in}$',extent=[0,2,0,100],aspect='0.2',vmin =np.min(conection_pos), vmax = np.max(conection_pos))
           #plt.colorbar(im1, orientation='vertical')
           #plt.xlim([-1,N_rec +1])
           #plt.ylim([-1,N_rec +1])
       
           plt.xticks(np.arange(0,3, 1))
           plt.yticks(np.arange(0,N_rec +1, 5))
           
           
           ## Wrec
           ax.append(fig2.add_subplot(1,3,2))
           plt.title(r'$W^{Rec}$', fontsize = 20)

           
       
           im=plt.imshow(conection_pos,interpolation='none',cmap=cmap,label='Conection matrix with', aspect="auto",vmin =np.min(conection_pos), vmax = np.max(conection_pos))
           #plt.colorbar(im, orientation='vertical')#,norm=norm_
           
           
       
           plt.xlim([-1,N_rec +1])
           plt.ylim([-1,N_rec +1])
       
           plt.xticks(np.arange(0,N_rec +1, 5))
           plt.yticks(np.arange(0,N_rec +1, 5))
           #plt.ylabel('Unit [i]',fontsize = 16)
           #plt.xlabel('Unit [j]',fontsize = 16)
           #plt.text(5, 5, 'Conection matrix with rank: '+str(rank)+'\n Det: '+str(deter), bbox={'facecolor': 'white', 'pad': 10})
           #plt.legend(fontsize= 'medium',loc=1)
           
           plt.ylabel('Post-synaptic',fontsize = 15) #Post-synaptic
           plt.xlabel('Pre-synaptic',fontsize = 15)
           plt.colorbar(im, orientation='vertical')
           
           ### W out
           ax.append(fig2.add_subplot(1,3,3))
           plt.title(r'$W^{out}$', fontsize = 20)
           im3=plt.imshow(out, cmap=cmap,interpolation="none",label= r'$W^{out}$',extent=[0,1,0,100],aspect='0.2',vmin =np.min(conection_pos), vmax = np.max(conection_pos))
           #plt.colorbar(im3, orientation='vertical')
           plt.xticks(np.arange(0,2, 1))
           plt.yticks(np.arange(0,N_rec +1, 5))
           
          
           
           fig2.tight_layout()
           
           plt.legend(fontsize= 'small',loc=1)
           plt.savefig(plot_dir+"/conection_matrix_P_"+str(i)+"_"+str(f)+"_"+str(string_name)+"_.png",dpi=200)
           plt.close()
           
           #################################################
           
           
           for ii in np.arange(0,10,1):
               a=y_train[ii, :, 0]
               b=y_pred[ii, :, 0]          
               a_min_b = np.linalg.norm(a-b)      
               lista_distancia.append(a_min_b)

       
           lista_distancia.insert(0,mem_gap)
           lista_distancia_all.append(lista_distancia)
          
           for sample_number in np.arange(sample_size_3):
           #   #if sample_number==2:#2#3xor
               print ("sample_number",sample_number)
               print_sample = plot_sample(sample_number,2,neurons,x_train,y_train,model,seq_dur,i,plot_dir,f,string_name)
               lista_freq_sample_net.append([string_name,sample_number,print_sample])
           print("print_sample",print_sample)

           #########################################
           fig= plt.figure(figsize=cm2inch(10,8.5))
           #fig.suptitle("\"And\" Data Set Trainined Output \n (amplitude in arb. units time in mSec)",fontsize = 20)
           for ii in np.arange(6):
               ab ="%.4f" % a_min_b
               a=y_train[ii, :, 0]
               b=y_pred[ii, :, 0]
               a_min_b = np.linalg.norm(a-b)      
               #lista_distancia.append(a_min_b)
               plt.subplot(3, 2, ii + 1)                
               plt.plot(x_train[ii+3, :, 0],color='g',label="Input A")
               #plt.plot(x_train[ii, :, 1],color='pink',label="Input B")
               #plt.plot(y_train[ii+3, :, 0],color='gray',linewidth=2,label="Expected Output")
               plt.plot(y_pred[ii+3, :, 0], color='r',label="Predicted Output\n Distance "+str(ab))
               #plt.ylim([-2, 1.6])
               plt.ylim([-2.5, 2])
               #plt.xlim([0, 205])
               #plt.xlim([0, 205])
               plt.xticks(np.arange(0,405,100),fontsize = 8)
               #plt.xticks(np.arange(0,405,100),fontsize = 8)
               #plt.legend(fontsize= 4.75,loc=3)
               leg = plt.legend(fontsize= 3.5,loc=3)
               leg.get_frame().set_linewidth(0.0)
               #plt.xticks([])
               plt.yticks([])
               plt.xticks(fontsize=4)
               plt.yticks(fontsize=4)
           fig.text(0.5, 0.03, 'time [mS]',fontsize=5, ha='center')
           fig.text(0.1, 0.5, 'Amplitude [Arb. Units]', va='center', ha='center', rotation='vertical', fontsize=5)
           figname =plot_dir+"/data_set_"+str(peso_mask)+'_'+str(mem_gap)+"_"+str(string_name)+"_.png"       
           plt.savefig(figname,dpi=300, bbox_inches = 'tight')

           K.clear_session()
todo_2       = np.c_[lista_distancia_all]
np.savetxt(plot_dir+'/distance_sample.txt',todo_2,fmt='%f %f %f %f %f %f %f %f %f %f %f',delimiter=' ',header="time_delay #S1 #S2 #S3 #S4 #S5 #S6 #S7 #S8 #S9 #S10")
#np.savetxt(plot_dir+'/freq_net.txt',net_freq,fmt='%s %s',delimiter='\t',header="Name #freq ")
#np.savetxt(plot_dir+'/freq_net_units.txt',lista_freq_sample_net,fmt='%s',delimiter='\t',header="Name #sample #freq ")
np.savetxt(plot_dir+'/H_number.txt',H_number_list,fmt='%s %s',delimiter='\t',header="time_delay #H number ")


print ("distancias",todo_2)
print("freq de las redes",net_freq)
print("freq unidades",lista_freq_sample_net)
