# Kreso_RNNs_work

Code, studies and preliminary results of the work about time encoding in different neuro-inspired tasks

For each task in this study, there is a folder including code and results (plots .png and .hdf5 files with RNNs weights to open with load_models_paper.py). To run the code for training networks you need to create locally a folder called "plots" and another called "weights" and adjust the path correctly for the code location. Be aware of the path of the input and output files. To generate the plots (meaning test the trained networks) it is necessary to create  "plots" output folder.

Code is written in python using:

- Matplotlib
- Numpy 
- Scipy
- Scikit learn
- Keras and Tensorflow

# Tasks considered so far are:

-  (1) Perceptual Decision making with time delay (analysis changing time of integration or delay)
-  (2) Perceptual Decision making with "Robust" trained with different sizes of integration time (analysis changing time of time delay)
-  (3) Perceptual Decision making with time ramp (analysis changing slope)
-  (4) Simple pulse copy with time Delay (analysis changing time delay)
-  (5) Output Ramp proportional to input amplitude (task Cue-Set-Go) from paper https://doi.org/10.1016/j.neuron.2022.12.016 (analysis changing slope)
-  (6) Perceptual Decision making "Long/short" depending on cue signal (21st June 2023)


# Note:
Code from task 6) includes the option of generating multiple trajectory plots. The path contains a file to generate the data set, which actually you can run to see how is the dataset used (generate_perceptual_dm), code to generate all plots of network activity (load_models_paper.py), which calls:  net_constraint_create.py and print_status_2_inputs_paper.py (or print_status_all_trayectories.py) depending on what you want to plot and also loop_call.py and recurrent_main_to_loop.py to train new networks in the task.
The directory contains a subdir called weights, which has inside the trained network that I showed today.
  
# Pending (currently working to implement):

- Context-dependent response time.
- Second task from the paper: Measure-Wait-Go (MWG).

# A draft made based on questions from everybody and some ideas (please comment or modify):

https://docs.google.com/document/d/1ASKTJqKTY3H4RIcuiNdQtsrRj3i2LvAJoFIwp3F3eG0/edit

# Code originally shared by Kreso:
https://github.com/afrojaakter/FallResearch2021/blob/main/vrnn_classifier_zero_entry_until_last_step.ipynb

# Ideas for the Analysis:

- Angle between trajectories. 
- Manifolds.
- To build the dynamical system using trajectories (trajectories of individual units are already available for the task considered)

