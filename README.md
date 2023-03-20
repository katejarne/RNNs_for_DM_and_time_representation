# Kreso_RNNs_work
Code, studies and preliminary results

For each task there is a folder including code and results (plots and .hdf5 files with RNNs weghts to open with load_models_paper.py). To run the code to train networks you need to create locally a folder called "plots" and other called "weights" and ajust the path correctly for the code location. Be aware of the path to the input and output files. To generate the plots it is necessary to create plots output folder.

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
- Output Ramp proporcional to input amplitude. (task Cue-Set-Go) from paper https://doi.org/10.1016/j.neuron.2022.12.016 ((analysis changing slope)

# Pending (currently working to implement):

- Context-dependent integration time.
- Context-dependent response time.
- Second task from paper: Measure-Wait-Go (MWG).


