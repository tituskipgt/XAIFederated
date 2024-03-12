import os
import argparse

import keras.utils
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import flwr as fl
from flwr.common import Metrics
from flwr.simulation.ray_transport.utils import enable_tf_gpu_growth
from os import listdir
from os.path import isfile, join
import pprint
from matplotlib import pyplot
import matplotlib.pyplot as plt

pp = pprint.PrettyPrinter(indent=4)
# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

parser = argparse.ArgumentParser(description="Flower Simulation with Tensorflow/Keras")

parser.add_argument(
    "--num_cpus",
    type=int,
    default=1,
    help="Number of CPUs to assign to a virtual client",
)
parser.add_argument(
    "--num_gpus",
    type=float,
    default=0.0,
    help="Ratio of GPU memory to assign to a virtual client",
)
parser.add_argument("--num_rounds", type=int, default=10, help="Number of FL rounds.")

VERBOSE = 0
TIME_STEPS = 4


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainset, testset, window_size, client_id) -> None:
        self.model = get_model(window_size)
        self.X_train = np.asarray(trainset[0])
        self.y_train = np.asarray(trainset[1])
        self.X_test = np.asarray(testset[0])
        self.y_test = np.asarray(testset[1])
        self.client_id = client_id

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        callback = tf.keras.callbacks.EarlyStopping()
        fit = self.model.fit(self.X_train, self.y_train, epochs=10, verbose=VERBOSE, validation_split=0.15)
        plot_performance(fit, self.client_id)
        return self.model.get_weights(), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss = self.model.evaluate(self.X_test, self.y_test, verbose=VERBOSE)

        return loss, len(self.X_test), {"mse": loss}


def preprocess_data_all(data_folder, time_steps):
    all_files = os.listdir(data_folder)

    # Filter only Excel files with a ".xlsx" extension
    file_names = [file for file in all_files if file.endswith(".xlsx")]
    file_names = file_names[:10]

    client_data = []
    for file_name in file_names:
        path = data_folder + '/' + file_name
        df = pd.read_excel(path, engine="openpyxl", nrows=20)

        scaler = MinMaxScaler()
        df['mg/dl'] = scaler.fit_transform(df['mg/dl'].values.reshape(-1, 1))

        sequences, labels = create_sequences(df['mg/dl'].values, time_steps)

        X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.15, shuffle=False)
        trains = [X_train, y_train]
        tests = [X_test, y_test]

        client = [trains, tests]
        client_data.append(client)

    return client_data


def preprocess_data(data_path):
    df = pd.read_excel(data_path, nrows=10)
    scaler = MinMaxScaler()
    df['mg/dl'] = scaler.fit_transform(df['mg/dl'].values.reshape(-1, 1))

    sequences, labels = create_sequences(df['mg/dl'].values, TIME_STEPS)

    X_train, X_val, y_train, y_val = train_test_split(sequences, labels, test_size=0.1, random_state=42)
    trainset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32)
    valset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(64)
    return trainset, valset


def create_sequences(data, time_steps):
    sequences, labels = [], []
    for i in range(len(data) - time_steps):
        seq = data[i:i + time_steps]
        label = data[i + time_steps]
        sequences.append(seq)
        labels.append(label)

    return np.array(sequences), np.array(labels)


def get_model(window_size):
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(64, input_shape=(window_size, 1)),
        tf.keras.layers.Dropout(0.005),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")

    return model


def get_client_fn(data, window_size):
    def client_fn(cid: str) -> fl.client.Client:
        client = data[int(cid)]
        trains = client[0]
        tests = client[1]
        return FlowerClient(trains, tests, window_size, int(cid)).to_client()

    return client_fn


def get_evaluate_fn(testset, window_size):
    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar], ):
        model = get_model(window_size)
        model.set_weights(parameters)
        loss = model.evaluate(testset, verbose=VERBOSE)[0]  # Extract MSE from evaluation result
        return loss, {"mse": loss}

    return evaluate


def mse_aggregation(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    mse_values = [m["mse"] for _, m in metrics]
    print("Values for MSE Global", end='\n')
    print(mse_values)
    return {"mse": np.mean(mse_values)}


def plot_performance(model, cid):
    pyplot.plot(model.history['loss'], label='train_loss')
    pyplot.plot(model.history['val_loss'], label='test_loss')
    title = 'Mean Square Error for Training and Test Dataset: Client #' + str(cid)
    pyplot.xlabel('Epochs')
    pyplot.ylabel('Mean Square Error')
    pyplot.title(title)
    pyplot.legend()
    pyplot.show()


def main() -> None:
    data_folder = 'Data'
    TIME_STEPS = 12
    client_data = preprocess_data_all(data_folder, TIME_STEPS)

    NUM_CLIENTS = len(client_data)

    # Create FedAvg strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.1,
        fraction_evaluate=0.05,
        min_fit_clients=5,
        min_evaluate_clients=5,
        min_available_clients=int(NUM_CLIENTS * 0.75),
        evaluate_metrics_aggregation_fn=mse_aggregation,
    )

    fl.simulation.start_simulation(
        client_fn=get_client_fn(client_data, TIME_STEPS),
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )


if __name__ == "__main__":
    enable_tf_gpu_growth()
    main()
