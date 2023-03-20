# Kreso_RNNs_work

Code, studies and preliminary results of the work about time encoding in different neuro inspired tasks

For each task in this study, there is a folder including code and results (plots .png and .hdf5 files with RNNs weights to open with load_models_paper.py). To run the code for training networks you need to create locally a folder called "plots" and another called "weights" and adjust the path correctly for the code location. Be aware of the path of the input and output files. To generate the plots (meaning test the trained networks) it is necessary to create  "plots" output folder.

Code is written in python using:

- Matplotlib
- Numpy 
- Scipy
- Scikit learn
- Keras and Tensorflow

# Task considered so far are:

- Perceptual Decision making with time delay (analysis changing time of integration or delay)
- Perceptual Decision making with "Robust" trained with different sizes of integration time (analysis changing time of time delay)
- Perceptual Decision making with time ramp ((analysis changing slope)
- Simple pulse copy with time Delay (analysis changing time delay)
- Output Ramp proportional to input amplitude. (task Cue-Set-Go) from paper https://doi.org/10.1016/j.neuron.2022.12.016 ((analysis changing slope)

# Pending (currently working to implement):

- Context-dependent integration time.
- Context-dependent response time.
- Second task from paper: Measure-Wait-Go (MWG).

# A draft made based on questions from everybothy and some ideas (please comment or modify):

https://docs.google.com/document/d/1ASKTJqKTY3H4RIcuiNdQtsrRj3i2LvAJoFIwp3F3eG0/edit

# Code originally sheared by Kreso:
https://github.com/afrojaakter/FallResearch2021/blob/main/vrnn_classifier_zero_entry_until_last_step.ipynb

# Ideas for the Analysis:

- Angle between trajectories. 

