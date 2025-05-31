from typing import List, Tuple, Dict

import numpy as np
from sklearn.linear_model import LogisticRegression
import json
import matplotlib.pyplot as plt
import os

from flwr.common import NDArrays, Metrics, Scalar



def get_model_parameters(model: LogisticRegression) -> NDArrays:
    """Return the parameters of a sklearn LogisticRegression model."""
    if model.fit_intercept:
        params = [
            model.coef_,
            model.intercept_,
        ]
    else:
        params = [
            model.coef_,
        ]
    return params




def set_model_params(model: LogisticRegression, params: NDArrays) -> LogisticRegression:
    """Set the parameters of a sklean LogisticRegression model."""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model


def set_initial_params(model: LogisticRegression, n_classes: int, n_features: int):
    """Set initial parameters as zeros.

    Required since model params are uninitialized until model.fit is called but server
    asks for initial parameters from clients at launch. Refer to
    sklearn.linear_model.LogisticRegression documentation for more information.
    """
    model.classes_ = np.array([i for i in range(n_classes)])

    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))

# This function is used at weighted average to store json file!
def store_metrics_to_json(metrics: Dict[str, float], filename: str = 'metrics.json'):
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
        



def plot_test_accuracy(filename='metrics.json'):
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
    test_accuracies = [item['test_accuracy'] for item in results if 'test_accuracy' in item]
    rounds = list(range(1, len(test_accuracies) + 1))

    # Plot the test accuracy over the rounds
    plt.figure(figsize=(8, 6))
    plt.plot(rounds, test_accuracies, linestyle='-', color='blue', label='Test Accuracy')
    plt.xlabel('Round')
    plt.ylabel('Test Accuracy')
    plt.title(f'Test Accuracy Over Rounds')
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

def delete_json_file(filename='metrics.json'):
        # Delete the JSON file after plotting
        try:
            os.remove(filename)
            print(f"{filename} has been successfully deleted.")
        except Exception as e:
            print(f"Error deleting {filename}: {e}")



def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Dict[str, Scalar]:
    """Compute weighted average.

    It is generic implementation that averages only over floats and ints and drops the
    other data types of the Metrics.
    """
    print(metrics)
    
    # num_samples_list can represent number of sample or batches depending on the client
    num_samples_list = [n_batches for n_batches, _ in metrics]
    num_samples_sum = sum(num_samples_list)
    metrics_lists: Dict[str, List[float]] = {}
    for num_samples, all_metrics_dict in metrics:
        #  Calculate each metric one by one
        for single_metric, value in all_metrics_dict.items():
            if isinstance(value, (float, int)):
                metrics_lists[single_metric] = []
        # Just one iteration needed to initialize the keywords
        break

    for num_samples, all_metrics_dict in metrics:
        # Calculate each metric one by one
        for single_metric, value in all_metrics_dict.items():
            # Add weighted metric
            if isinstance(value, (float, int)):
                metrics_lists[single_metric].append(float(num_samples * value))

    weighted_metrics: Dict[str, Scalar] = {}
    for metric_name, metric_values in metrics_lists.items():
        weighted_metrics[metric_name] = sum(metric_values) / num_samples_sum

    store_metrics_to_json(weighted_metrics) #store to json
    return weighted_metrics


# def store_client_metrics(data, filename='results.json'):
       
#         try:
#             # Load the existing data
#             try:
#                 with open(filename, 'r') as file:
#                     file_data = json.load(file)
#                     if not isinstance(file_data, list):
#                         file_data = [file_data]  # Convert to list if it's a dictionary
#             except FileNotFoundError:
#                 file_data = []

#             # Append new data
#             file_data.append(data)

#             # Write the updated data back to the file
#             with open(filename, 'w') as file:
#                 json.dump(file_data, file, indent=4)

#         except Exception as e:
#             print(f"Error updating JSON file: {e}")

        