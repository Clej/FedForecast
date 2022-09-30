from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from util.utils import loss_torch_to_skl, train_one_step, test_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
from approach.lstm_forecast import ShallowForecastLSTM

def set_zero_weight_model(model):
    """Set all the parameters of a model to 0"""

    for layer_weigths in model.parameters():
        layer_weigths.data.sub_(layer_weigths.data)

# FedAvg functions
def FedAvg_agregation_process(model, clients_params, weight_client_loss, device):
    """Creates the server model by averaging clients' model's params"""

    central_model = deepcopy(model).to(device)

    set_zero_weight_model(central_model)

    for k, params_k in enumerate(clients_params):

        for idx, layer_weights in enumerate(central_model.named_parameters()):

            # average all parameters except biases ?
            if 'bias' or 'linear' in layer_weights[0]:
                continue

            contribution = params_k[idx].to(device).data * weight_client_loss[k]
            layer_weights.data.add_(contribution)

    print('Fed averaging step done !')

    return central_model

def FedAvg_loop(train_sets, test_sets, call_basemodel, archi_basemodel={'n_inputs':1, 'n_hidden': 10}, n_cr=50, n_local_epochs=1, device='cpu', **kwargs_optim):
    """FevAvg main loop"""

    # number of clients
    K = len(train_sets)
    assert K == len(test_sets) # as many clients as test sets

    # model architecture
    n_inputs = archi_basemodel.get('n_inputs')
    n_hidden = archi_basemodel.get('n_hidden') # number of neurons in the hidden lay

    # constant model initialization across clients' model
    ## to copy the architecture and parameters value
    # server_model = deepcopy(base_model).to(device)
    server_model = call_basemodel(num_var=n_inputs, hidden_units=n_hidden)

    # each model is initialized with same weights
    models = [call_basemodel(num_var=n_inputs, hidden_units=n_hidden) for _ in range(K)]

    # training loss function
    loss_function = nn.MSELoss()

    # inti each loach optimizer
    lr_i = kwargs_optim.get('lr', 1e-2)
    decay = kwargs_optim.get('decay_lr', 0.95)
    local_optimizers = [torch.optim.SGD(models[k].parameters(), lr=lr_i) for k in range(K)]
    # local_schedulers = [ReduceLROnPlateau(local_optimizers[k], 'min', patience = 3) for k in range(K)]

    # a client weight is proportional to its sample size
    n_tot_samples = sum([len(train_set_i) for train_set_i in train_sets])
    weight_clients_loss = np.array([len(train_set_i) / n_tot_samples for train_set_i in train_sets])

    # keep track of local losses along communication rounds
    server_loss_it = []
    clients_loss_it_bfr_fedavg = []
    clients_loss_it_aft_fedavg = []

    # for each communication round
    for i in range(n_cr):

        print(f'Communication round {i}\n-------------------')
        local_params = []

        # compute initial local losses
        client_losses_bfr_fedavg = np.empty(shape=(K,))
        client_losses_aft_fedavg = np.empty_like(client_losses_bfr_fedavg)

        # compute server loss: weighted mean loss across clients
        server_loss = np.average(client_losses_bfr_fedavg, weights=weight_clients_loss)

        # for each client
        for k in range(K):
            
            print(f'**Learning client#{k}\n--------------')

            # compute local testing losses before FedAvg step
            # model_k = deepcopy(models[k])
            client_losses_bfr_fedavg[k] = test_model(
                test_sets[k],
                models[k],
                loss_function,
                verbose=False
            )

            # initialize local models with the current server_model weights
            # model_k = deepcopy(server_model).to(device)
            # local_optim_k = None
            # local_optim_k = local_optimizers[k]

            # perform gradient step(s) on local data from server_model weights
            for _ in range(n_local_epochs):

                _, models[k] = train_one_step(
                    data_loader=train_sets[k],
                    model=models[k],
                    loss_function=loss_function,
                    optimizer=local_optimizers[k]
                )

            # compute local testing losses after FedAvg step
            client_losses_aft_fedavg[k] = test_model(
                test_sets[k],
                models[k],
                loss_function,
                verbose=False
            )

            # keep track of the weights of the local models
            local_params_k = list(models[k].parameters())
            local_params_k = [tensor_params.detach() for tensor_params in local_params_k]
            local_params.append(local_params_k)

            # local learning rate scheduler
            # local_schedulers[k].step(test_model(test_sets[k], models[k], loss_function))

            # models[k] = deepcopy(model_k)
            # model_k = None # release memory

            print(f'**Client test loss before FedAvg {client_losses_bfr_fedavg[k]}')
            print(f'**Client test loss after FedAvg {client_losses_aft_fedavg[k]}')
            print('\n--------------')
            #--

        # weighted average of client model parameters
        server_model = FedAvg_agregation_process(
            deepcopy(server_model),
            local_params,
            device=device,
            weight_client_loss=weight_clients_loss
            )
        
        # keep track of global loss
        server_loss = np.average(client_losses_aft_fedavg, weights=weight_clients_loss)

        # keep track of the losses w.r.t communication rounds
        clients_loss_it_aft_fedavg.append(client_losses_aft_fedavg)
        clients_loss_it_bfr_fedavg.append(client_losses_bfr_fedavg)
        server_loss_it.append(server_loss)

        # exponential decay of local learning rate
        # lr_i *= decay

        print(f'Global test loss = {server_loss}\n')
    #--
    return models, clients_loss_it_bfr_fedavg, clients_loss_it_aft_fedavg, server_loss_it