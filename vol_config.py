import torch
import numpy as np
import pandas as pd

class bcolors:
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    PINK = '\033[35m'
    GREY = '\033[36m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

np.set_printoptions(formatter={'float': '{: 0.4f}'.format})

# number of runs
n_runs = 20
n_stock = 5
seq_len = 4
n_iter = 2000
batch = 10

torch.manual_seed(12345)
# check gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

rng = np.random.default_rng(12345)

# f = vol_obj(in_size=seq_len, out_size=1).to(device)


returns = pd.read_csv('Returns.csv')


def return_sample(start_idx, batch, seq_len):
    result = np.zeros((batch, seq_len, n_stock))
    for idx in range(batch):
        val_ = returns.iloc[(start_idx+idx):(start_idx+idx+seq_len)][['MMM', 'MSFT', 'JPM', 'AMZN', 'XOM']].values
        result[idx, :, :] = val_
    return result