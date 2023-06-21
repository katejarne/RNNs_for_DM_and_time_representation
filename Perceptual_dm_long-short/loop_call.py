##########################################################
#                 Author C. Jarne                        #
#               call loop  (ver 1.0)                     #                       
# MIT LICENCE                                            #
##########################################################

import os
import time
#import numpy as np
from recurrent_main_to_loop import *

#custom init and constraint (dale's law) if needed
 

start_time = time.time()
vector=np.arange(1)


f          ='weights'
f_plot     ='plots'

distancias = []

for t in vector:
    #for i in np.arange(10,180,10):
    for i in [10]:
        mem_gap = i
        N_rec   =150
        base= f+'/'+  os.path.basename(f+'_'+str(mem_gap)+'_N_'+str(N_rec)+'_gap_'+str(i))
        base_plot= f_plot+'/'+  os.path.basename(f_plot+'_'+str(t)+'_N_'+str(i))
        dir = str(base)
        if not os.path.exists(dir):
           os.mkdir(base)
        print(str(dir))

        dir = str(base_plot)
        if not os.path.exists(dir):
           os.mkdir(base_plot)        
        print(str(dir))
        #cont= constrain(N_rec)
        pepe    =DM_fun(mem_gap,N_rec,base,base_plot)
        distancias.append(pepe)
print('-------------------------')
print (distancias)
print("--- %s to train the network seconds ---" % (time.time() - start_time))
