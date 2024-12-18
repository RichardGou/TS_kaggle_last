import pandas as pd
import gzip
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def create_windows(data, input_length, output_length, stride=None):
    if stride is None:
        stride = input_length + output_length 
    X, y = [], []
    for i in range(0, len(data) - input_length - output_length + 1, stride):
        X.append(data[i:i + input_length])
        y.append(data[i + input_length:i + input_length + output_length])
    return np.array(X), np.array(y)


def get_traffic_data(path_data, stride, batch_size=64, output_length=24):
    df = pd.read_csv(path_data, header=None)

    time_series = df.iloc[:, 0].values
    input_length = 168  # Points passés (1 semaine)
    output_length = 24  # Points futurs (24 heures)
    train_split = 0.6
    valid_split = 0.2

    n_points = len(time_series)
    train_end = int(train_split * n_points)
    valid_end = int((train_split + valid_split) * n_points)

    # Training
    train_data = time_series[:train_end]
    X_train, y_train = create_windows(train_data, input_length, output_length, stride=stride)

    # Validation
    valid_data = time_series[train_end:valid_end]
    X_valid, y_valid = create_windows(valid_data, input_length, output_length, stride=stride)

    # Test
    test_data = time_series[valid_end:]
    X_test, y_test = create_windows(test_data, input_length, output_length, stride=stride)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
    X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32).unsqueeze(-1)
    y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32).unsqueeze(-1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(TensorDataset(X_valid_tensor, y_valid_tensor), batch_size=batch_size, drop_last=True)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size, drop_last=True)
    
    return train_loader, valid_loader, test_loader