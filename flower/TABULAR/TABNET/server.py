import flwr as fl
import json
import matplotlib.pyplot as plt #add to env
import os
import tensorflow as tf
import tabnet
from tensorflow import keras
from flwr.common import ndarrays_to_parameters
from flwr.common import NDArrays, Metrics, Scalar
from typing import List, Tuple, Dict
from flwr.server.strategy import DifferentialPrivacyServerSideAdaptiveClipping

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from datasets import Dataset, DatasetDict
from datasets import load_dataset
from sklearn.model_selection import train_test_split
#pip install --upgrade flwr
#might have dependencies conflict with protobuf, use protobuf <4


if __name__ == "__main__":
    num_rounds = 300
    client_var = 100
    dataset_name = 'Cancer'+' dataset' # change name for plot

    # stores metric in json
    def store_metrics_to_json(metrics: Dict[str, float], filename: str = 'metrics_tabnet.json'):
        try:
            # Load existing metrics if the file exists
            with open(filename, 'r') as file:
                data = json.load(file)
        except FileNotFoundError:
            data = []

        # Append new metrics
        data.append(metrics)

        # Write the updated metrics back to the file
        with open(filename, 'w') as file:
            json.dump(data, file, indent=4)

    def plot_accuracy(filename='metrics_tabnet.json'):
        """
        Plot test accuracy from the JSON file. 

        Args:
            filename (str): Path to the JSON file containing the evaluation results.
        """
        try:
            # Load evaluation results from the JSON file
            with open(filename, 'r') as file:
                results = json.load(file)
        except Exception as e:
            print(f"Error reading JSON file: {e}")
            return

        # Extract test accuracies and identify the rounds in the sequence
        accuracies = [item['accuracy'] for item in results if 'accuracy' in item]
        rounds = list(range(1, len(accuracies) + 1))

        # Plot the test accuracy over the rounds
        plt.figure(figsize=(8, 6))
        plt.plot(rounds, accuracies, linestyle='-', color='red', label='Accuracy')
        plt.xlabel('Round')
        plt.ylabel('Accuracy TabNet')
        plt.title(f'{dataset_name} Accuracy Over {num_rounds} Rounds {client_var} Clients')
        plt.grid(True)
        plt.legend()
        # Set ticks every `tick_step` rounds
        if max(rounds) <= 200:
            tick_interval = 10
        elif 200 < max(rounds) <= 500:
            tick_interval = 50
        else:
            tick_interval = 100
        plt.xticks(range(0, max(rounds) + 1, tick_interval))

        plt.show()

    def delete_json_file(filename='metrics_tabnet.json'):
            # Delete the JSON file after plotting
            try:
                os.remove(filename)
                print(f"{filename} has been successfully deleted.")
            except Exception as e:
                print(f"Error deleting {filename}: {e}")



    # Define metric aggregation function
    def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        # Multiply accuracy of each client by number of examples used
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]

        store_metrics_to_json({"accuracy": sum(accuracies) / sum(examples)})
        # Aggregate and return custom metric (weighted average)
        return {"accuracy": sum(accuracies) / sum(examples)}

##############TO GET INITIAL PARAMETERS FOR STRATEGIES THAT NEED IT#############################
    feature_columns = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
       'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave_points_worst',
       'symmetry_worst', 'fractal_dimension_worst']
    
    lst_columns = []
    for col_name in feature_columns:
        lst_columns.append(tf.feature_column.numeric_column(col_name))
    # Load TabNet model
    model = tabnet.TabNetClassifier(
        lst_columns,
        num_classes=2, # change
        feature_dim=8,
        output_dim=4,
        num_decision_steps=4,
        relaxation_factor=1.0,
        sparsity_coefficient=1e-5,
        batch_momentum=0.98,
        virtual_batch_size=None,
        norm_type="group",
        num_groups=1,
    )
    lr = tf.keras.optimizers.schedules.ExponentialDecay(
        0.01, decay_steps=100, decay_rate=0.9, staircase=False
    )
    optimizer = tf.keras.optimizers.Adam(lr)
    model.compile(optimizer, loss="categorical_crossentropy", metrics=["accuracy"])    
    ####################################################################################################
    
    
    # add strategies
    strategy = fl.server.strategy.FedAvg(
            min_available_clients=client_var,
            min_fit_clients = client_var,
            min_evaluate_clients = client_var,
            # fit_metrics_aggregation_fn=weighted_average,
            evaluate_metrics_aggregation_fn=weighted_average,
            # initial_parameters = fl.common.ndarrays_to_parameters(model.get_weights())
        )  

    med_strategy = fl.server.strategy.FedAvg(
            min_available_clients=client_var,
            min_fit_clients = client_var,
            min_evaluate_clients = client_var,
            # fit_metrics_aggregation_fn=weighted_average,
            evaluate_metrics_aggregation_fn=weighted_average,
        )  

    prox_strategy = fl.server.strategy.FedProx( 
        evaluate_metrics_aggregation_fn=weighted_average,
        min_fit_clients= client_var,
        min_evaluate_clients= client_var,
        min_available_clients=client_var,
        proximal_mu =2.0 #float!
    )

    # yogi_strategy = fl.server.strategy.FedYogi(
    #     evaluate_metrics_aggregation_fn=weighted_average,
    #     min_fit_clients=client_var,
    #     min_evaluate_clients=client_var,
    #     min_available_clients=client_var,
    #     initial_parameters=ndarrays_to_parameters(initial_parameters)
    # )


     # DP
    dp_strategy = DifferentialPrivacyServerSideAdaptiveClipping(
        strategy,
        noise_multiplier=0.1,
        # clipping_norm=10,
        num_sampled_clients=client_var,
        clipped_count_stddev = 1
    )


    # Start Flower server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy =strategy, #added
        config=fl.server.ServerConfig(num_rounds=num_rounds),
    )

    # wipes metric file after every run!!!
    plot_accuracy()
    delete_json_file()
