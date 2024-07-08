import argparse
import warnings

import os
import flwr as fl
import tensorflow as tf
import tensorflow_datasets as tfds
import tabnet


from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from datasets import Dataset, DatasetDict
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from flwr.client.mod import fixedclipping_mod
from flwr.client.mod.localdp_mod import LocalDpMod 


# Read before starting. Use dataset with headers, and make sure there is no space in the columns names
# CHANGE N_client & num_partitions in client.py  & STRATEGY & min_available_clients in server.py
# Preferably use dataset where label is at [0] and feature from 1:. You can modify code nevertheless

if __name__ == "__main__":
    BATCH_SIZE = 50
    N_CLIENTS = 100 # CHANGE THIS & num_partitions & STRATEGY & min_available_clients
    num_class = 2 # CHANGE THIS TOO 
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

    # Make TensorFlow log less verbose
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


    #dataset with headers, make sure it is shuffled
    dataset_path = '/Users/allan/dataset/cancer_test_headers.csv'

    # Load the dataset using datasets from Hugging Face's library
    dataset = load_dataset('csv', data_files=dataset_path)
    partitioner = IidPartitioner(num_partitions= N_CLIENTS) #CHANGE THIS
    partitioner.dataset = dataset['train']
    partition = partitioner.load_partition(partition_id).with_format("pandas")[:] #change to id later, takes partition of certain id
    X = partition.iloc[:, 1: ] #needed?
    y = partition.iloc[:,0 ] #needed?
    unique_labels = y.unique() #needed?

    # Assuming column names are known and the last column is the label
    feature_columns = partition.columns[1:]  # All columns except the first one as features
    label_column = partition.columns[0]      # first column as label


    train_df, test_df = train_test_split(partition, test_size=0.2, random_state=42)

    def df_to_dataset(dataframe, shuffle=False, batch_size=32):
        df = dataframe.copy()
        labels = df.pop(label_column)
        ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
        if shuffle:#shuffles partition
            ds = ds.shuffle(buffer_size=len(dataframe))
        ds = ds.batch(batch_size)
        return ds

    ds_train = df_to_dataset(train_df, batch_size=50)
    ds_test = df_to_dataset(test_df, batch_size=50)

    def transform(features, labels):
        # Assume labels are categorical and need to be converted to one-hot
    
        y = tf.one_hot(labels, depth=len(partition[label_column].unique()))  # Set depth to the number of unique labels
        return features, y

    lst_columns = []
    for col_name in feature_columns:
        lst_columns.append(tf.feature_column.numeric_column(col_name))
    ds_train = ds_train.map(transform)
    ds_test = ds_test.map(transform)


    # Make TensorFlow log less verbose
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Load TabNet model
    model = tabnet.TabNetClassifier(
        lst_columns,
        num_classes=num_class, # change
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
    
    # Define Flower client
    class TabNetClient(fl.client.NumPyClient):
        def get_parameters(self, config):
            return model.get_weights()

        def fit(self, parameters, config):
            model.set_weights(parameters)
            model.fit(ds_train, epochs=25)
            
            return model.get_weights(), len(ds_train), {}



        def evaluate(self, parameters, config):
            model.set_weights(parameters)
            loss, accuracy = model.evaluate(ds_test) #change name
            return loss, len(ds_test), {"accuracy": accuracy}

    def client_fn(cid: str):
            return IrisClient().to_client()

    # #Add fixedclipping_mod to the client-side mods
    # app = fl.client.ClientApp(
    #     client_fn=client_fn,
    #     mods=[
    #         fixedclipping_mod,
    #     ]
    # )


    # Start Flower client
    fl.client.start_client(
        server_address="127.0.0.1:8080", client=TabNetClient().to_client()
    )

    