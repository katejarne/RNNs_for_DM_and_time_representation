# Task-Parametrised Dynamics: How Recurrent Networks encode time and decision making through structured connectivity

The repository contains code for training Recurrent Neural Networks (RNNs), visualising the trained networks (in terms of connectivity or network activity in response to different stimuli), and conducting various analyses. Each function is well documented.

Dependencies:

tensorflow → 2.18.0

matplotlib → 3.6.3

numpy → 1.26.4

scipy → 1.11.4

scikit-learn → 1.4.1

h5py → 3.12.1

networkx → 3.4.2

# Code originally shared by Kreso:
https://github.com/afrojaakter/FallResearch2021/blob/main/vrnn_classifier_zero_entry_until_last_step.ipynb

**Training Networks:**
1. To train a set of networks for a specific task, refer to the tasks listed in Table 1 of the paper (Below). Additional tasks are also available in the repository.

| Task                                      | Elapsed time represented with       | Figure | Label                                                                                          | Number of time intervals | Integration                                      |
|-------------------------------------------|--------------------------------------|--------|------------------------------------------------------------------------------------------------------------------|--------------------------|--------------------------------------------------|
| Simple Delayed Binary Decision Making     | None                                 | 1c     | Simple DM| 1                        | No                                               |
| Context-dependent Binary Decision Making  | Pulse Amplitude                      | 1d     | Simple DM Long-short                                                                 | 2                        | No                                               |
| Multi-interval Amplitude-based Decision   | Pulse Amplitude                      | 5      | Simple DM 8 times/ 4 times                                                                                                            | 8                        | No                                               |
| Multi-interval Distance-based Decision    | Pulse distance                       | 1e     | Simple DM 8 time encoded                                                                                                              | 8                        | No                                               |
| Time interval comparison task (TICT)      | Comparison between intervals         | Sup.   | Interval compare                                                                                         | 1                        | No                                               |
| Windowed Integration Decision Making      | None                                 | 1f     | Integral DM       | 1                        | During a fixed window before the decision        |
| Fixed Integration time Decision Making    | None                                 | 1g     | Integral DM signal keep        | 1                        | During a fixed window (signal continues after)   |
| Cued Integration time Decision Making     | Cue and Pulse amplitude              | 1h     | Integral DM Cue    | 1                        | During a fixed window before decision (continues)|

To train a set of networks, execute the script [`train_loop_main.py`](train_loop_main.py). Specify how many networks you would like to train for the chosen task (vector) and the number of recurrent units (N_rec).
This script invokes the function [`recurrent_main_to_train_loop.py`](recurrent_main_to_train_loop.py), where you can set various training parameters. By selecting the corresponding task label, the standard parameters for that task will be loaded. This function calls one of the dataset generators located in the "data_set_generators" directory. The script will create directories (one for each network), containing the iterations of the network's 20 training epochs and a file named `100_final.hdf5`, which represents the final trained network. These directories are saved in the "weights" directory.

2. **Visualising Network Responses:**
   The repository includes results from the trained networks for each studied task. You can visualise the response of any network to stimuli by opening its corresponding dataset generator using the function [`load_RNNs_models_to_plot.py`](load_RNNs_models_to_plot.py). The results will be saved in the "plots" directory.

3. **Creating Animations:**
   To create an animation of the network behaviour for two selected tasks—Context-dependent Binary Decision Making and Windowed Integration Decision Making—use the script[`animation_for_rnn_model.py`](animation_for_rnn_model.py). The output will be a GIF.

4. **Analysing Generalised Correlations:**
   To examine the generalised correlation of input weights with the input stimulus, run the script [`generalized_correlation_with_input.py`](generalized_correlation_with_input.py). To analyse the generalised correlation of network activity with the output, execute [`generalized_correlation_with_output.py`](generalized_correlation_with_output.py)  and select the appropriate network. These scripts support the same tasks mentioned above (Context-dependent Binary Decision Making and Windowed Integration Decision Making), but they can be modified for any task.

5. **Estimating Principal Eigenvalues:**
   To estimate the principal eigenvalues of all networks stored in the results (the weights folder), run the script [`itera_eigen.py`](itera_eigen.py). There are also functions within the "[`plot_utilities`](plot_utilities)" directory that can be used to visualise the results generated by this script. These functions start with `plot_eigen*py`.

6. **Studying Network Connectivity and Properties:**
   To analyse the network’s connectivity and properties, run the script  [`itera_network_node_conn_in_rnn.py`](itera_network_node_conn_in_rnn.py) and select the network of interest.

7. **Calculating the Sequentiality Index:**
   Use the script [`intera_SI.py`](intera_SI.py) to calculate the sequentiality index.

Finally, the "[`plot_utilities`](plot_utilities)" directory contains scripts to generate visualisations for all tasks mentioned in the paper, as well as scripts for visualising normality results (`plot_normality`), the sequentiality index (`plot_SI_task_compare`), and other functions. Each script is accompanied by a description.
