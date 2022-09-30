"""Train/test of a simple LSTM based time series forecasting """
import numpy as np

from matplotlib import pyplot as plt

from statsmodels.tsa.tsatools import lagmat

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from torch.utils.data import DataLoader
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam

from approach.lstm_forecast import ShallowForecastLSTM
from util.utils import loss_torch_to_skl, train_one_step, test_model, predict, SequenceDataset

# import data: you need to download it, see readme.md in the datasets directory
# x_data = np.load("datasets/traffic.npy")
T_size, p = x_data.shape
T_train = int(0.70 * T_size)

lag = 40
p_sub = 1 # reduced number of time series components

# scale the data
scaler =  StandardScaler(with_mean=True, with_std=True)

train_dataset = SequenceDataset(
    scaler.fit_transform(x_data[:T_train, :p_sub]), lag=lag
    )
test_dataset = SequenceDataset(
    scaler.transform(x_data[T_train:, :p_sub]), lag=lag
    )

batch_size = 2**3
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# print('X :')
# next(iter(test_loader))[0]

# print('Y :')
# next(iter(test_loader))[1]

# desgin the model
num_hidden_units = 50
model = ShallowForecastLSTM(num_var=train_dataset.p, hidden_units=num_hidden_units)

loss_torch_to_skl(model, train_loader, mean_squared_error)

#==========================
# train the model
loss_function = nn.MSELoss()
learning_rate = 1e-2
optimizer = Adam(model.parameters(), lr=learning_rate, betas=(0.6, 0.75))
scheduler = ReduceLROnPlateau(optimizer, 'min', patience = 5)

print("Untrained test\n--------")
test_model(test_loader, model, loss_function)

n_epoch = 5*10**1

train_loss_epoch = np.empty(n_epoch)
test_loss_epoch = np.empty(n_epoch)

for i_epoch in range(n_epoch):
    print(f"Epoch {i_epoch}\n---------")

    train_loss_epoch[i_epoch] = train_one_step(
        train_loader, model, loss_function, optimizer=optimizer
        )[0]
    test_loss_epoch[i_epoch] = test_model(test_loader, model, loss_function)
    scheduler.step(test_loss_epoch[i_epoch])
#==========================

# visualize train/test loss trajectories
plt.plot(np.arange(n_epoch), train_loss_epoch, label='train loss')
plt.plot(np.arange(n_epoch), test_loss_epoch, label='test loss')
plt.legend(); plt.show()

# visualize the predictions and actual values
t_test = np.arange(T_train , T_size)
x_test_pred = scaler.inverse_transform(predict(test_loader, model).numpy())

t_train = np.arange(0, T_train)
train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
x_train_pred = scaler.inverse_transform(
    predict(train_eval_loader, model).numpy()
    )

# index of a variable of the time series
idx_p = 0
# visualize forecasting on training set
plt.subplot(211)
plt.plot(t_train, x_data[:T_train, idx_p], marker='o', label='Truth', color='C0', alpha=0.25)
plt.plot(t_train, x_train_pred[:, idx_p], color='red', label='Prediction', alpha=0.25)

plt.legend()
# plt.show()

# visualize forecasting on testing set
plt.subplot(212)
plt.plot(t_test, x_data[T_train:, idx_p], marker='o', color='C0', alpha=0.25)
plt.plot(t_test, x_test_pred[:, idx_p], color='red', alpha=0.25)
plt.xlabel('time')
plt.show()

#----- draft
# a = np.arange(1, 7+1)
# lagmat(a, maxlag=3, trim='both', original='sep')

# for layer_weight in model.named_parameters():
#     print("\n Layer {}: ".format(layer_weight[0]))
#     print("shape of layer: {}".format(layer_weight[1].shape))
#     print(layer_weight)


# for X, y in enumerate(test_loader):
#     print(f'X')