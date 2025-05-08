# itera_network_node_conn_in_rnn.py
"""
MIT License - C. Jarne V. 1.0 - 2025
This code systematically analyzes the functional and structural
impact of individual neuron removal in RNNs performing delayed
decision-making tasks. It iteratively disconnects each neuron
by zeroing its incoming/outgoing connections, then evaluates
how this ablation affects: 1) Prediction accuracy through
output-target distance metrics. 2) Network topology properties
(clustering, efficiency, path lengths). The implementation
generates comparative visualizations of connectivity matrices,
input-output mappings, and complex network measure trends.
"""
import os
import fnmatch
import tensorflow as tf
import networkx as nx
from plot_utilities.utilities import cm2inch
from data_set_generators.generate_DM_delayed_response_sample import *

# Define paths
current_directory = os.path.dirname(__file__)
# r_dir = current_directory + "/weights/DM_delayed_response/weights_20_N_100_gap_0/"
r_dir = current_directory + "/weights/01_DM_delayed_response/orthogonal_rrn_no_bias_term/weights_20_N_100_gap_0/"
# r_dir = current_directory+"/weights/05_Perceptual_dm_delayed_response/large_out_weights/weights_N_100_2/"
plot_dir = "plots"

# Other constants
N_rec = 100
sample_size = 50
mem_gap = 10
x_train, y_train, seq_dur = generate_trials(sample_size, mem_gap)
test_set = x_train[0:20, :, :]
y_test_set = y_train[0:20, :, 0]

# Load and configure model
for root, sub, files in os.walk(r_dir):
    files = sorted(files)
    r_dir = root
    folder_name = os.path.basename(root)
    string_name = str(folder_name)
    for i, f in enumerate(files):
        if fnmatch.fnmatch(f, '*100_final.hdf5'):
            model = tf.keras.models.load_model(r_dir + f, compile=False)
            model.compile(loss='mse', optimizer='Adam')
            complex_network_measures = []
            distance_list_all = []

            weights_in = model.layers[0].get_weights()[0]
            weights_out = model.layers[1].get_weights()[0]
            connection = model.layers[0].get_weights()[1]

            # Original configuration
            original_connection = connection.copy()
            model.layers[0].set_weights([weights_in, original_connection])

            # Prediction for the full connected network
            y_pred_original = model.predict(x_train[0:10, :, :])

            # Calculate average reference distance
            distances_original = [np.linalg.norm(y_train[ii, :, 0] - y_pred_original[ii, :, 0]) for ii in range(10)]
            avg_distance_original = np.mean(distances_original)

            # Calculate reference complex network metrics
            G_original = nx.from_numpy_array(original_connection > 0)
            complex_measures_original = {
                "clustering": nx.average_clustering(G_original),
                "efficiency": nx.global_efficiency(G_original),
                "shortest_path": nx.average_shortest_path_length(G_original) if nx.is_connected(G_original) else np.nan,
                "assortativity": nx.degree_assortativity_coefficient(G_original),

            }

            # Iterate over each neuron, setting its row and column to zero
            for neuron_idx in range(N_rec):
                modified_connection = connection.copy()
                modified_connection[neuron_idx, :] = 0
                modified_connection[:, neuron_idx] = 0

                model.layers[0].set_weights([weights_in, modified_connection])

                # Prediction with modified connectivity matrix
                y_pred = model.predict(x_train[0:10, :, :])

                # Plot connectivity matrices and results
                fig, axs = plt.subplots(1, 3, figsize=(15, 5))

                axs[0].imshow(weights_in.T, cmap="seismic",interpolation="none", label=r'$W^{in}$',
                              extent=[0, 1, 0, 100], aspect='0.2')
                axs[0].set_title("W_in")
                axs[1].imshow(modified_connection, cmap="seismic", aspect="1")
                axs[1].set_title(f"Modified W_rec (neuron {neuron_idx} disconnected)")
                axs[2].imshow(weights_out, cmap="seismic", interpolation="none", label=r'$W^{out}$',
                              extent=[0, 1, 0, 100], aspect='0.2')
                axs[2].set_title("W_out")
                axs[1].set_xticks(np.arange(0, N_rec + 1, 10))
                axs[1].set_yticks(np.arange(0, N_rec + 1, 10))
                axs[1].set_ylabel('Post-synaptic', fontsize=15)
                axs[1].set_xlabel('Pre-synaptic', fontsize=15)
                plt.savefig(f"{plot_dir}/modified_Wrec_neuron_{neuron_idx}_{f}.png", dpi=200)
                plt.close(fig)

                # Calculate and save the average distance metric for this configuration
                distances = [np.linalg.norm(y_train[ii, :, 0] - y_pred[ii, :, 0]) for ii in range(10)]
                avg_distance = np.mean(distances)

                # Plot comparison between prediction and ground truth

                distance_list = []
                for ii in range(10):
                    a = y_train[ii, :, 0]
                    b = y_pred[ii, :, 0]
                    a_min_b = np.linalg.norm(a - b)
                    distance_list.append(a_min_b)
                avg_distance = np.mean(distance_list)
                distance_list_all.append(avg_distance)

                fig = plt.figure(figsize=cm2inch(10, 8.5))
                fig.suptitle(f"unit {neuron_idx} unconnected")
                for ii in np.arange(6):
                    a = y_train[ii, :, 0]
                    b = y_pred[ii, :, 0]
                    a_min_b_ = np.linalg.norm(a - b)
                    ab = "%.4f" % a_min_b_

                    plt.subplot(3, 2, ii + 1)
                    plt.plot(x_train[ii, :, 0], color='g', label="Input")
                    if x_train.shape[2] == 2:
                        plt.plot(x_train[ii, :, 1],color='pink',label="Cue")
                    plt.plot(y_train[ii, :, 0], color='gray', linewidth=2, label="Expected Output")
                    plt.plot(y_pred[ii, :, 0], color='r', linewidth=1,
                             label="Predicted Output\n Distance " + str(ab))

                    plt.ylim([-2.5, 2.5])
                    plt.xticks(np.arange(0, len(y_train[ii+3])+5, 50), fontsize=8)
                    leg = plt.legend(fontsize=3.5, loc=3)
                    leg.get_frame().set_linewidth(0.0)
                    plt.yticks([])
                    plt.xticks(fontsize=4)
                    plt.yticks(fontsize=4)
                fig.text(0.5, 0.03, 'time [ms]', fontsize=5, ha='center')
                fig.text(0.1, 0.5, 'Amplitude [Arb. Units]', va='center', ha='center',
                         rotation='vertical', fontsize=5)
                plt.savefig(f"{plot_dir}/comparison_neuron_{neuron_idx}_{f}.png", dpi=200)
                plt.close(fig)

                network_matrix = modified_connection
                G = nx.from_numpy_array(network_matrix > 0)  # Convert to binary adjacency matrix

                # Compute complex network measures
                clustering_coefficient = nx.average_clustering(G)
                efficiency = nx.global_efficiency(G)
                try:
                    avg_shortest_path = nx.average_shortest_path_length(G)
                except nx.NetworkXError:
                    avg_shortest_path = np.nan  # If the graph is disconnected

                # Additional complex network measures
                assortativity = nx.degree_assortativity_coefficient(G)
                degree_centrality = np.mean(list(nx.degree_centrality(G).values()))
                closeness_centrality = np.mean(list(nx.closeness_centrality(G).values()))
                try:
                    eccentricity = np.mean(list(nx.eccentricity(G).values()))
                except nx.NetworkXError:
                    eccentricity = np.nan  # If the graph is disconnected
                try:
                    diameter = nx.diameter(G)
                except nx.NetworkXError:
                    diameter = np.nan  # If the graph is disconnected
                try:
                    radius = nx.radius(G)
                except nx.NetworkXError:
                    radius = np.nan  # If the graph is disconnected

                # Store results
                complex_network_measures.append({
                    "neuron_idx": (neuron_idx,neuron_idx),
                    "clustering_coefficient": clustering_coefficient,
                    "efficiency": efficiency,
                    "avg_shortest_path": avg_shortest_path,
                    "assortativity": assortativity,
                    "degree_centrality": degree_centrality,
                    "closeness_centrality": closeness_centrality,
                    "eccentricity": eccentricity,
                    "diameter": diameter,
                    "radius": radius
                })

            # Plot distance vs interval
            plt.figure(figsize=(10, 6))
            # Línea horizontal para red completa
            plt.axhline(y=avg_distance_original,  color='r', linestyle='--',
                        linewidth=2, label=f'Undamaged network ({avg_distance_original: .2f})')
            ymax = max(max(distance_list_all), avg_distance_original) * 1.1
            # plt.plot(range(len(distance_list_all)), distance_list_all, marker='o', linestyle='-')
            plt.plot(np.arange(N_rec), distance_list_all, marker='o', linestyle='-',
                     label="Removing each unit from the x-axis")

            plt.title("Average Distance (predicted-target) vs Unit id")
            plt.xlabel("Unit id")
            plt.ylabel("Average Distance")
            # plt.axhline(2, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
            plt.xticks(ticks=np.arange(0, 101, 10))
            # plt.grid(True)
            plt.legend()
            plt.ylim([0, ymax])
            plt.savefig(plot_dir + "/" + str(folder_name) + f"Average_Distance_vs_threshold_{i}.png")

            # Plot network properties
            # interval_labels = [f"{round(m['interval'][0], 2)}-{round(m['interval'][1], 2)}" 
            # for m in complex_network_measures]

            # Clustering coefficient
            plt.figure(figsize=(10, 6))
            # plt.plot(interval_labels, [m['clustering_coefficient'] for m in complex_network_measures],
            # marker='o', linestyle='-', color='b', label="Clustering Coefficient")
            plt.plot(np.arange(N_rec), [m['clustering_coefficient'] for m in complex_network_measures],
                     marker='o', linestyle='-', color='b', label="Clustering Coefficient")
            plt.xlabel("Threshold")
            plt.ylabel("Clustering Coefficient")
            plt.title("Clustering Coefficient vs Interval")
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.savefig(plot_dir + "/" + str(folder_name) + f"Clustering_Coefficient_vs_Interval_{i}.png")

            # Efficiency
            plt.figure(figsize=(10, 6))
            plt.plot(np.arange(N_rec), [m['efficiency'] for m in complex_network_measures], marker='o',
                     linestyle='-', color='g', label="Efficiency")
            # plt.xlabel("Interval")
            plt.xlabel("Threshold")
            plt.ylabel("Global Efficiency")
            plt.title("Global Efficiency vs Threshold")
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.savefig(plot_dir + "/" + str(folder_name) + f"Efficiency_vs_Interval_{i}.png")

            # Average shortest path length
            plt.figure(figsize=(10, 6))
            plt.plot(np.arange(N_rec), [m['avg_shortest_path'] for m in complex_network_measures], marker='o',
                     linestyle='-', color='r', label="Average Shortest Path")
            # plt.xlabel("Interval")
            plt.xlabel("Threshold")
            plt.ylabel("Average Shortest Path Length")
            plt.title("Average Shortest Path vs Threshold")
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.savefig(plot_dir + "/" + str(folder_name) + f"Average_Shortest_Path_{i}.png")

            # Assortativity
            plt.figure(figsize=(10, 6))
            plt.plot(np.arange(N_rec), [m['assortativity'] for m in complex_network_measures], marker='o',
                     linestyle='-', color='purple')
            plt.xlabel("Threshold")
            plt.ylabel("Assortativity")
            plt.title("Assortativity vs Threshold")
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.savefig(plot_dir + "/" + str(folder_name) + f"Assortativity_{i}.png")

            # Diameter
            plt.figure(figsize=(10, 6))
            plt.plot(np.arange(N_rec), [m['diameter'] for m in complex_network_measures], marker='o',
                     linestyle='-', color='magenta')
            plt.xlabel("Threshold")
            plt.ylabel("Diameter")
            plt.title("Diameter vs Threshold")
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.savefig(plot_dir + "/" + str(folder_name) + f"diameter_{i}.png")

            # Radius
            plt.figure(figsize=(10, 6))
            plt.plot(np.arange(N_rec), [m['radius'] for m in complex_network_measures], marker='o',
                     linestyle='-', color='grey')
            plt.xlabel("Threshold")
            plt.ylabel("Radius")
            plt.title("Radius vs Threshold")
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.savefig(plot_dir + "/" + str(folder_name) + f"radius_{i}.png")

            # Degree centrality
            plt.figure(figsize=(10, 6))
            plt.plot(np.arange(N_rec), [m['degree_centrality'] for m in complex_network_measures], marker='o',
                     linestyle='-', color='orange')
            plt.xlabel("Threshold")
            plt.ylabel("Degree Centrality")
            plt.title("Degree Centrality vs Threshold")
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.savefig(plot_dir + "/" + str(folder_name) + f"degree_centrality_{i}.png")

            # Closeness centrality
            plt.figure(figsize=(10, 6))
            plt.plot(np.arange(N_rec), [m['closeness_centrality'] for m in complex_network_measures], marker='o',
                     linestyle='-', color='brown')
            plt.xlabel("Threshold")
            plt.ylabel("Closeness Centrality")
            plt.title("Closeness Centrality vs Threshold")
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.savefig(plot_dir + "/" + str(folder_name) + f"degree_closeness_{i}.png")

            # Eccentricity
            plt.figure(figsize=(10, 6))
            plt.plot(np.arange(N_rec), [m['eccentricity'] for m in complex_network_measures], marker='o',
                     linestyle='-', color='cyan')
            plt.xlabel("Threshold")
            plt.ylabel("Eccentricity")
            plt.title("Eccentricity vs Threshold")
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.savefig(plot_dir + "/" + str(folder_name) + f"degree_eccentricity_{i}.png")

            # Print complex network measures for verification
            for i, measure in enumerate(complex_network_measures):
                # print(f"Interval {measure['interval']}:")
                print(f" - Clustering Coefficient: {measure['clustering_coefficient']}")
                print(f" - Efficiency: {measure['efficiency']}")
                print(f" - Average Shortest Path Length: {measure['avg_shortest_path']}")
                print(f" - Assortativity: {measure['assortativity']}")
                print(f" - Degree Centrality: {measure['degree_centrality']}")
                print(f" - Closeness Centrality: {measure['closeness_centrality']}")
                print(f" - Eccentricity: {measure['eccentricity']}")
                print(f" - Diameter: {measure['diameter']}")
                print(f" - Radius: {measure['radius']}")

