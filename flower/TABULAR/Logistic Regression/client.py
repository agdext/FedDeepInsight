import argparse
import warnings

from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, roc_curve, auc
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import json


import flwr as fl
import utils
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from datasets import Dataset, DatasetDict
from datasets import load_dataset
from flwr.client.mod.localdp_mod import LocalDpMod 
from flwr.client.mod import fixedclipping_mod

if __name__ == "__main__":
    N_CLIENTS = 20 # !CHANGE THIS & num_partitions & STRATEGY & min_available_clients!

    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--partition-id",
        type=int,
        choices=range(0, N_CLIENTS),
        required=True,
        help="Specifies the artificial data partition",
    )
    args = parser.parse_args()
    partition_id = args.partition_id

    
    # dataset_path = '/Users/allan/dataset/stroke_balancedH.csv'
    dataset_path = '/Users/allan/dataset/cancer_test_headers.csv'

    # Load the dataset using datasets from Hugging Face's library
    dataset = load_dataset('csv', data_files=dataset_path)
    partitioner = IidPartitioner(num_partitions=N_CLIENTS) #CHANGE THIS
    partitioner.dataset = dataset['train']
    partition = partitioner.load_partition(partition_id).with_format("pandas")[:]
    X = partition.iloc[:, 1: ]
    y = partition.iloc[:, 0 ]
    unique_labels = y.unique()

    # Split the on edge data: 80% train, 20% test
    X_train, X_test = X[: int(0.8 * len(X))], X[int(0.8 * len(X)) :]
    y_train, y_test = y[: int(0.8 * len(y))], y[int(0.8 * len(y)) :]

    # Create LogisticRegression Model
    model = LogisticRegression(
        penalty="l2",
        max_iter=1,  # local epoch
        warm_start=True,  # prevent refreshing weights when fitting
    )

    # Setting initial parameters, akin to model.compile for keras models
    utils.set_initial_params(model, n_features=X_train.shape[1], n_classes=2)

    # Method for extra learning metrics calculation
    def eval_learning(y_test, y_pred):
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(
            y_test, y_pred, average="micro"
        )  # average argument required for multi-class
        prec = precision_score(y_test, y_pred, average="micro")
        f1 = f1_score(y_test, y_pred, average="micro")
        return acc, rec, prec, f1
    

    

    # Define Flower client
    class IrisClient(fl.client.NumPyClient):
        round_count = 0  # Initialize the round count

        def get_parameters(self, config):  # type: ignore
            return utils.get_model_parameters(model)

        def fit(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)
            # Ignore convergence failure due to low local epochs
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
            accuracy = model.score(X_train, y_train)
            return (
                utils.get_model_parameters(model),
                len(X_train),
                {"train_accuracy": accuracy},
            )

        def evaluate(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)
            loss = log_loss(y_test, model.predict_proba(X_test), labels=unique_labels)
            accuracy = model.score(X_test, y_test)
            
            y_pred = model.predict(X_test)
            acc, rec, prec, f1 = eval_learning(y_test, y_pred)

            # Increment round count before saving
            IrisClient.round_count += 1

            output_dict = {
                "round": IrisClient.round_count,
                "accuracy": accuracy,  
                "acc": acc,
                "loss":loss,
                "rec": rec,
                "prec": prec,
                "f1": f1,
            }

            #utils.store_client_metrics(output_dict, 'evaluation_per_client.json') 
            
            return loss, len(X_test), {"test_accuracy": accuracy}
            
        
    def client_fn(cid: str):
            return IrisClient().to_client()
        

    # # Create an instance of the mod with the required params
    # local_dp_obj = LocalDpMod(clipping_norm=10, 
    #                           sensitivity=1, 
    #                           epsilon=0.000001,
    #                           delta=1
    # )
    # # Add local_dp_obj to the client-side mods

    # app = fl.client.ClientApp(
    # client_fn=client_fn,
    # mods=[local_dp_obj],
    # )

    

#     #Add fixedclipping_mod to the client-side mods
#     app = fl.client.ClientApp(
#         client_fn=client_fn,
#         mods=[
#             fixedclipping_mod,
#         ]
# )

    # Start Flower client
    fl.client.start_client(
        server_address="0.0.0.0:8080", client=IrisClient().to_client()
    )
# poetry run python3 server.py &
# poetry run python3 client.py --partition-id 0 &
# poetry run python3 client.py --partition-id 1 &
# poetry run python3 client.py --partition-id 2 &
# poetry run python3 client.py --partition-id 3 &
# poetry run python3 client.py --partition-id 4

# poetry run python3 server.py &
# poetry run python3 client.py --partition-id 0 &
# poetry run python3 client.py --partition-id 1 &
# poetry run python3 client.py --partition-id 2 &
# poetry run python3 client.py --partition-id 3 &
# poetry run python3 client.py --partition-id 4 &
# poetry run python3 client.py --partition-id 5 &
# poetry run python3 client.py --partition-id 6 &
# poetry run python3 client.py --partition-id 7 &
# poetry run python3 client.py --partition-id 8 &
# poetry run python3 client.py --partition-id 9 &
# poetry run python3 client.py --partition-id 10 &
# poetry run python3 client.py --partition-id 11 &
# poetry run python3 client.py --partition-id 12 &
# poetry run python3 client.py --partition-id 13 &
# poetry run python3 client.py --partition-id 14 &
# poetry run python3 client.py --partition-id 15 &
# poetry run python3 client.py --partition-id 16 &
# poetry run python3 client.py --partition-id 17 &
# poetry run python3 client.py --partition-id 18 &
# poetry run python3 client.py --partition-id 19
