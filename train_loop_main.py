"""
Author C. Jarne
loop  call (ver 1.0) 2024

Script to create directories and run a main loop to train
RNNs with specified recurrent units. Each iteration
calls a custom function `DM_fun` with parameters to train
a model, saving results (distances) and model weights/plots
in dynamically created directories.

"""

from recurrent_main_to_train_loop import *
import time

start_time = time.time()
vector = np.arange(0, 10, 1)
f = 'weights'
f_plot = 'plots'

distancias = []

for t in vector:
    # for i in np.arange(10,180,10):
    for i in [20]:
        # mem_gap = i
        N_rec = 100  # 200
        base = f+'/'+os.path.basename(f+'_N_'+str(N_rec)+'_'+str(t))
        base_plot = f_plot+'/'+os.path.basename(f_plot+'_'+str(t)+'_N_'+str(t))
        dir = str(base)
        if not os.path.exists(dir):
           os.mkdir(base)
        print(str(dir))
        dir = str(base_plot)
        if not os.path.exists(dir):
           os.mkdir(base_plot)        
        print(str(dir))
        # cont= constrain(N_rec)
        pepe = DM_fun(mem_gap, N_rec, base, base_plot)
        distancias.append(pepe)
print('-------------------------')
print(distancias)
print("--- %s to train the network seconds ---" % (time.time() - start_time))
