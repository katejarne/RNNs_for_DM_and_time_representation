"""
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
# pca part:
from sklearn.decomposition import PCA

def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

def plot_sample(sample_number,input_number,neurons,x_train,y_train,model,seq_dur,i,plot_dir,f,string_name,mem_gap):
    seq_dur                        = len(x_train[sample_number, :, 0])
    test                           = x_train[sample_number:sample_number+1,:,:]
    capa=0

    #Primer capa:
    get_0_layer_output = K.function([model.layers[capa].input], [model.layers[capa].output])
    
    layer_output= get_0_layer_output([test])[capa]
        
    #segunda capa:
    get_1_layer_output = K.function([model.layers[capa].input], [model.layers[capa].output])
    layer_output_T       = layer_output.T

    print("layer_output",layer_output_T)
    array_red_list       = []

    # To generate the Populational Analysis

    for ii in np.arange(0,neurons,1):
        neurona_serie = np.reshape(layer_output_T[ii], len(layer_output_T[ii]))
        array_red_list.append(neurona_serie)
    
    array_red = np.asarray(array_red_list)
    pca       = PCA(n_components=3)
    X_pca_    = pca.fit(array_red)
    X_pca     = pca.components_
    print("------------")
    ordeno_primero_x=X_pca[0]
    ordeno_primero_y=X_pca[1]
    ordeno_primero_z=X_pca[2]

    kk=70
    fig = plt.figure(figsize=cm2inch(19,7))
    ax  = fig.add_subplot(122, projection='3d')

    x=X_pca[0]
    y=X_pca[1]
    z=X_pca[2]
    N=len(z)

    for ik in range(N-1):
        ax.plot(x[ik:ik+2], y[ik:ik+2], z[ik:ik+2], color=plt.cm.viridis(ik/N))
    ax.scatter(ordeno_primero_x[0],ordeno_primero_y[0],ordeno_primero_z[0],s=70,c='r',marker="^",label=' Start ')     
    ax.scatter(ordeno_primero_x[-1],ordeno_primero_y[-1],ordeno_primero_z[-1],s=70,c='b',marker="^",label=' Stop ')

    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    ax.set_zticks(())
    ax.view_init(elev=10, azim=kk)
    ax.legend(fontsize= 6)

    figname = str(plot_dir)+"/"+str(mem_gap)+"_Sample_"+str(sample_number)+"_pca_3D__individual_neurons_state_"+str(i)+'_'+str(kk)+"_"+str(f)+"_"+str(string_name)+".png"
    plt.savefig(figname,dpi=300, bbox_inches = 'tight') 
    plt.close()      
    ####################################
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from keras import backend as K
# pca part:
from sklearn.decomposition import PCA

def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

def plot_samples(sample_numbers, input_number, neurons, x_train, y_train, model, seq_dur, plot_dir, f, string_name, mem_gap):
    fig = plt.figure(figsize=cm2inch(12, 10))
    ax = fig.add_subplot(122, projection='3d')

    for i, sample_number in enumerate(sample_numbers):
        seq_dur = len(x_train[sample_number, :, 0])
        test = x_train[sample_number:sample_number+1, :, :]

        capa = 0

        # Primer capa:
        get_0_layer_output = K.function([model.layers[capa].input], [model.layers[capa].output])
        layer_output = get_0_layer_output([test])[capa]

        # Segunda capa:
        get_1_layer_output = K.function([model.layers[capa].input], [model.layers[capa].output])
        layer_output_T = layer_output.T

        array_red_list = []

        # To generate the Populational Analysis
        for ii in np.arange(0, neurons, 1):
            neurona_serie = np.reshape(layer_output_T[ii], len(layer_output_T[ii]))
            array_red_list.append(neurona_serie)

        array_red = np.asarray(array_red_list)
        pca = PCA(n_components=3)
        X_pca_ = pca.fit(array_red)
        X_pca = pca.components_

        ordeno_primero_x = X_pca[0]
        ordeno_primero_y = X_pca[1]
        ordeno_primero_z = X_pca[2]

        kk = 70

        x = X_pca[0]
        y = X_pca[1]
        z = X_pca[2]
        N = len(z)

        for ik in range(N-1):
            ax.plot(x[ik:ik+2], y[ik:ik+2], z[ik:ik+2], color=plt.cm.viridis(ik/N),linewidth=0.2,alpha=0.25)

        ax.scatter(ordeno_primero_x[0], ordeno_primero_y[0], ordeno_primero_z[0], s=5, c='r', marker="^", label=' Start ')
        ax.scatter(ordeno_primero_x[-1], ordeno_primero_y[-1], ordeno_primero_z[-1], s=5, c='b', marker="^", label=' Stop ')

    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    ax.set_zticks(())
    #ax.set_xlim([np.min(ordeno_primero_x), np.max(ordeno_primero_x)])
    #ax.set_ylim([np.min(ordeno_primero_y), np.max(ordeno_primero_y)])
    #ax.set_zlim([np.min(ordeno_primero_z), np.max(ordeno_primero_z)])
    ax.view_init(elev=10, azim=kk)
    #ax.legend(fontsize=6)

    figname = f"{plot_dir}/{mem_gap}_pca_3D_individual_neurons_state_{string_name}.png"
    plt.savefig(figname, dpi=300, bbox_inches='tight')
    #plt.close()
    plt.show()
