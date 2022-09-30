import torch
from torch.utils.data import Dataset

# Class to format train/test datasets
class SequenceDataset(Dataset):
    def __init__(self, data_np, lag=48):
        self.lag = lag
        # number of components in the time-series
        self.is_univts = data_np.ndim == 1
        self.p = data_np.shape[1] if data_np.ndim == 2 else 1
        self.X = torch.from_numpy(data_np.reshape(-1, self.p)).float()

    def __len__(self):
        # the last timestep has no target value, 
        # so we can only train on n-1 timesteps
        # or n-h timesteps if we wanted to learn forecasting h-step-ahead
        return self.X.shape[0]

    def __getitem__(self, i): 
        if i >= self.lag - 1*0:
            i_start = i - self.lag + 1*0
            x = self.X[i_start:(i + 1*0)]
        else:
            padding = self.X[0].repeat(self.lag - i - 1*0, 1)
            x = self.X[0:(i + 1*0)]
            x = torch.cat((padding, x), 0)

        # before that, self.X is always a 2d tensor
        if self.is_univts:
            return x, self.X[i].ravel()
        else:
            return x, self.X[i]

# train loop
def train_one_step(data_loader, model, loss_function, optimizer):
    """
    perform a training step on a data_loader batch,
    and returns both average loss on training batches and model
    """

    num_batches = len(data_loader)
    total_loss = 0.0
    
    model.train()
    for X, y in data_loader:
        output = model(X)
        loss = loss_function(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    print(f"Train loss: {avg_loss}")

    return avg_loss, model

# test loop
def test_model(data_loader, model, loss_function, verbose=True):
    """return average loss on test batches"""

    num_batches = len(data_loader)
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            output = model(X)
            total_loss += loss_function(output, y).item()

    avg_loss = total_loss / num_batches
    if verbose:
        print(f"Test loss: {avg_loss}")

    return avg_loss
#

def predict(data_loader, model):
    """predict from torch model and data loader inputs"""

    output = torch.tensor([])

    model.eval()
    with torch.no_grad():
        for X, _ in data_loader:
            y_hat = model(X)
            output = torch.cat((output, y_hat), 0)

    return output
#

def loss_torch_to_skl(model, data_loader, metric_skl, **kwargs_metric):
    """Compute prediction error from torch model with an sklearn metric"""

    y_hat = torch.tensor([])
    y_true = torch.empty_like(y_hat)
    sample_weight = kwargs_metric.get('weight_client', None)

    model.eval()
    with torch.no_grad():
        # for each batch
        for X, y in data_loader:
            u = model(X)
            y_hat = torch.cat((y_hat, u))
            y_true = torch.cat((y_true, y))

    y_true, y_hat = y_true.numpy(), y_hat.numpy()
    return metric_skl(y_true, y_hat, sample_weight=sample_weight) #* y_true.size
#