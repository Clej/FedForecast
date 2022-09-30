Application of Federated Averging algorithm to train LSTM models for time series forecasting. Notebook is written in french.

## Libraries

Libraries used in this repo:

sklearn: 0.24.2
pytorch: 1.9.0
numpy: 1.19.5
matplotlib: 3.2.2

## Datasets

See datasets directory to get the datasets used. In the scripts and notebook, a dataset is a multivariate time-series stored in a numpy array with size ``(T_size, n_variables)``, where ``T_size`` is the number of time steps (same along the time series components) and ``n_variables`` is the dimension of a single temporal sample.

## Author

Clément Lejeune.

## References

Datasets from:
H. F. Yu, N. Rao, and I. S. Dhillon, “Temporal regularized matrix factorization for high-dimensional time series prediction,” in NIPS, 2016, pp. 847–855.

Federated Averaging algorithm:
H. B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. Agüera y Arcas, “Communication-Efﬁcient Learning of Deep Networks from Decentralized Data,” in AISTATS, 2017, vol. 54.
