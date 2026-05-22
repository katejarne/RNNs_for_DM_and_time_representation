# Full code for the paper: "Task-Parametrized Dynamics: Representation of Time and Decisions in Recurrent Neural Networks"

[`https://www.biorxiv.org/content/10.1101/2025.09.15.676356v1`](https://www.biorxiv.org/content/10.1101/2025.09.15.676356v1) 

Abstract:
How do recurrent neural networks (RNNs) internally represent elapsed time to initiate responses after learned delays? To address this question, we trained RNNs on delayed decision-making tasks {with progressively increasing temporal demands, including binary decisions, context-dependent decisions, and perceptual integration. We analyzed trained networks using connectivity statistics, eigenvalue spectra, readout alignment, and low-dimensional population trajectories. Across tasks, networks converged to qualitatively distinct but behaviourally comparable dynamical solutions, including oscillatory and non-oscillatory (ramping/decaying) regimes, consistent with solution degeneracy. Population activity remained low-dimensional and distributed across recurrent units rather than localized to individual neurons. Readout alignment was strongly epoch-dependent: activity evolved largely in the readout-null subspace prior to response generation and became increasingly aligned with the output dimension near decision time. In sign-symmetric tasks, trained networks preserved an approximate sign-flip equivariance inherited from architecture and training symmetry, despite independent noisy perturbations across trials, yielding mirrored population responses across stimulus sign. Together, these results show that temporal and decision-related computations can emerge through multiple dynamical regimes, while maintaining structured low-dimensional representations and comparable behavioural performance, mirroring biological principles of degeneracy and functional redundancy.


The repository contains code for training Recurrent Neural Networks (RNNs), visualising the trained networks (in terms of connectivity or network activity in response to different stimuli), and conducting various analyses. Each function is well documented.

Dependencies:

tensorflow → 2.18.0

matplotlib → 3.6.3

numpy → 1.26.4

scipy → 1.11.4

scikit-learn → 1.4.1

h5py → 3.12.1

networkx → 3.4.2

**Training Networks:**
1. To train a set of networks for a specific task, refer to the tasks listed in Table 1 of the paper (Below). Additional tasks are also available in the repository.

| Task Name                           | Parameter varied          | Figure          | Reference                                      | # Intervals | Label in Code           | Integration                                |
| :---------------------------------- | :------------------------ | :-------------: | :--------------------------------------------: | :---------: | :---------------------: | :----------------------------------------- |
| **Simple Delayed Binary Decision Making** | Stimulus sign           | Fig 1c        | [Stanislaw, 1999]                            | 1           | Simple DM               | No                                         |
| **Context-dependent Binary Decision Making** | Stimulus amplitude     | Fig 1d        | [Mante et al., 2013]                          | 2           | Simple DM Long-short    | No                                         |
| **Multi-interval Amplitude-based Decision Making** | Pulse Amplitude    | SI Fig 1      | N/A                                           | 8           | Simple DM 8 times/4 times | No                                         |
| **Multi-interval Distance-based Decision Making** | Pulse distance      | 1e & SI Fig 2 | N/A                                           | 8           | Simple DM 8 time encoded | No                                         |
| **Time interval comparison task (TICT)** | Comparison between intervals | SI Fig 3     | [Diaz et al., 2025]                           | 1           | Interval compare        | No                                         |
| **Windowed Evidence Integration (Perceptual Decision Making)** | None                | Fig 1f        | [Newsome et al., 1989; Roitman et al., 2002; Kiani et al., 2008] | 1 | Integral DM      | During fixed window before decision        |
| **Continuous Evidence Integration** | None               | Fig 1g        | [Newsome et al., 1989; Roitman et al., 2002; Kiani et al., 2008] | 1 | Integral DM signal keep | During fixed window (signal continues after) |
| **Cued Evidence Integration** | Cue and Pulse amplitude | Fig 1h       | [Newsome et al., 1989; Roitman et al., 2002; Kiani et al., 2008] | 1 | Integral DM Cue | During fixed window before decision (continues) |

*Table: Summary of all tasks included in the study.*  
*Category 1: Simple decisions with single delay*  
*Category 2: Binary decisions with multiple intervals*  
*Category 3: Perceptual integration tasks*


To train a set of networks, execute the script [`train_loop_main.py`](train_loop_main.py). Specify how many networks you would like to train for the chosen task (vector) and the number of recurrent units (N_rec).
This script invokes the function [`recurrent_main_to_train_loop.py`](recurrent_main_to_train_loop.py), where you can set various training parameters. By selecting the corresponding task label, the standard parameters for that task will be loaded. This function calls one of the dataset generators located in the ["data_set_generators"](data_set_generators) directory. The script will create directories (one for each network), containing the iterations of the network's 20 training epochs and a file named `100_final.hdf5`, which represents the final trained network. These directories are saved in the "weights" directory. Please create a "weights" directory in the project folder.

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
   Use the script [`itera_SI.py`](itera_SI.py) to calculate the sequentiality index.

The "[`plot_utilities`](plot_utilities)" directory contains scripts to generate visualisations for all tasks mentioned in the paper, as well as scripts for visualising normality results [`plot_normality.py`](`plot_utilities/plot_normality.py`), the sequentiality index [`plot_SI_task_compare.py`](`plot_utilities/plot_SI_task_compare.py`), and other functions. Each script is accompanied by a description.

# Pipeline Overview: RNN Dynamics Classification

The pipeline in the `network_clasification_papeline` folder trains and analyzes recurrent neural networks (RNNs) on the different timing tasks, classifying each network’s internal dynamics into **Oscillatory**, **Ramping/Decaying**, or **Mixed** regimes using a dual spectral‑activity criterion.

[`run_experiment.py`](./network_clasification_papeline/run_experiment.py) is the top‑level launcher. It iterates over predefined tasks (e.g., “Simple Delayed Binary DM”) and weight initializations (`Normal` / `Orthogonal`). For each combination, it calls [`train_ensemble.py`](./network_clasification_papeline/train_ensemble.py), which repeatedly trains independent RNNs (10 replicas by default) until a low training MSE is reached. Trained models are saved together with loss curves and sample predictions in structured subfolders (e.g., `results/Simple_DM_Long-short_Normal/net_00/`).

After training, [`analyze_ensemble_v4.py`](./network_clasification_papeline/analyze_ensemble_v4.py) processes every saved network. It computes a spectral oscillation index (OI) from the recurrent weight matrix’s dominant eigenvalue, and an activity‑based metric: the log10 Peak‑to‑Background Ratio (PBR) and a ramp index (absolute Pearson correlation of PC1 with time). The dual classification requires both criteria to agree for a pure regime; otherwise, the network is labelled `Mixed`. The script generates eigenvalue plots, PSD figures, PC1 traces, and summary tables (CSV, text, LaTeX). Finally, [`recompute_accuracy.py`](./network_clasification_papeline/recompute_accuracy.py) can be run afterwards to recalculate classification accuracy on a fixed test set using a robust response‑window method that excludes neutral trials, and it updates the per‑network statistics files.

This work is based on the idea from  Kresimir Josic to study time encoding # Code:
https://github.com/afrojaakter/FallResearch2021/blob/main/vrnn_classifier_zero_entry_until_last_step.ipynb

