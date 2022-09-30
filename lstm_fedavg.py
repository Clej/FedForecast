"""Train/test of multiple LSTM based time forecasting models with FedAvg"""
import numpy as np

from matplotlib import pyplot as plt

from sklearn.preprocessing import StandardScaler

from torch.utils.data import Dataset, DataLoader

from approach.lstm_forecast import ShallowForecastLSTM
from util.utils import train_one_step, test_model, predict, SequenceDataset
from util.fed_utils import FedAvg_loop

# import data: you need to download it, see readme.md in the datasets directory
# x_data = np.load("datasets/traffic.npy")
T_size, p = x_data.shape
T_train = int(0.70 * T_size)

lag = 48
n_clients = 2 # reduced number of time series components

# scale the data
scalers =  [StandardScaler(with_mean=True, with_std=True) for _ in range(n_clients)]

batch_size = 2**3

train_loaders, test_loaders = [], []

for k in range(n_clients):
    train_loaders.append(
        DataLoader(
            SequenceDataset(
                scalers[k].fit_transform(x_data[:T_train, k][:, np.newaxis]),
                lag=lag
            ),
            batch_size=batch_size, shuffle=True
        )
    )

    test_loaders.append(
        DataLoader(
            SequenceDataset(
                scalers[k].transform(x_data[T_train:, k][:, np.newaxis]),
                lag=lag
            ),
            batch_size=batch_size, shuffle=False
        )
    )

# Instantiate a base pytorch model
# base_model = ShallowForecastLSTM(num_var=1, hidden_units=25)

# train with FedAvg
models, local_losses_bfr_fedavg, local_losses_aft_fedavg, global_loss  = FedAvg_loop(
    call_basemodel=ShallowForecastLSTM, # give a CALL TO THE model instances, not the instances
    archi_basemodel={'n_inputs':1, 'n_hidden': 10},
    train_sets=train_loaders,
    test_sets=test_loaders,
    n_cr=5, n_local_epochs=5,
    lr= 10**-2
)

np.array(local_losses_bfr_fedavg)

for k in range(n_clients):
    plt.plot(np.array(local_losses_bfr_fedavg)[:, k], label='loss before FedAvg', alpha=0.4)
    # plt.plot(local_losses_aft_fedavg[k], label='loss before FedAvg', alpha=0.4)
plt.show()
