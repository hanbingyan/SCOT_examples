# Volatility calibration
import torch
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from datetime import datetime
from nets import mv_obj
from configs import *
from utils import *


def train(f, idx):

    var_hist = torch.zeros(n_iter, device='cpu')

    f_mean = torch.zeros(n_iter, device='cpu')
    weights = torch.zeros((n_iter, n_stock), device='cpu')

    optimF = optim.Adam(list(f.parameters()), lr=1e-2)

    x = return_sample(start_idx=idx, batch=batch, seq_len=seq_len)
    x = torch.tensor(x, dtype=torch.float, device=device)

    for iter in range(n_iter):

        fx = f(x).mean()
        f.zero_grad()
        fx.backward()
        optimF.step()

        with torch.no_grad():
            f.weight.clamp_(1e-5, 1.0)
            # self.transition.data.div_(torch.sum(model.transition.data, dim=0))
            # f.weight.div_(torch.sum(f.weight.data)) #= f.weight.data/f.weight.data.sum()
            f.weight += (1 - f.weight.data.sum()) / n_stock

            # if iter % 100 == 0:
            #     for name, param in f.named_parameters():
            #         print(name, param.data)
                # print('Theoretical return', (1+torch.dot(f.weight.data, torch.tensor([0.1, 0.0, 0.0, 0.0, -0.1],
                #                                                                      device=device)))**5 - 1)

            # for param in f.parameters():
            #     param.clamp_(0.0, 1.0)

        f_mean[iter] = f(x).mean().item()
        weights[iter, :] = f.weight.data.cpu()

        if iter % 200 == 0:
            print('iter', iter, 'dual', var_hist[iter].item(), 'f(y)', f(x).mean().item())


    return f_mean.numpy(), weights.numpy()

############## Main #######################
if __name__ == '__main__':

    for risk_aver in [0.0, 0.1, 1.0]:
        print(bcolors.GREEN + 'Current risk aversion:', risk_aver, bcolors.ENDC)

        f_runs = []
        NonRtn_runs = []
        NonWeight_runs = []
        NaiveRtn_runs = []

        for scene in range(n_runs):

            scene = scene*10

            f = mv_obj(weight_init=torch.ones(n_stock)/n_stock, mean_init=torch.ones(1),
                       risk_aversion=risk_aver).to(device)

            f_mean, weights = train(f, scene)


            out_of_sample_idx = scene + batch + seq_len
            last_weights = weights[-10:, :].mean(axis=0)
            out_of_sample_x = return_sample(out_of_sample_idx, batch=1, seq_len=seq_len)
            out_of_sample_rtn = np.matmul(out_of_sample_x, last_weights)
            out_of_sample_rtn = np.prod(1 + out_of_sample_rtn)
            print('Nonrobust weights:', last_weights)
            print('Nonrobust return (+1.0):', out_of_sample_rtn)

            NonWeight_runs.append(last_weights)
            NonRtn_runs.append(out_of_sample_rtn)

            naive_weights = np.ones(n_stock)/n_stock
            naive_rtn = np.matmul(out_of_sample_x, naive_weights)
            naive_rtn = np.prod(1 + naive_rtn)
            print('Naive return (+1.0):', naive_rtn)

            NaiveRtn_runs.append(naive_rtn)

            f_runs.append(f_mean)


        f_runs = np.array(f_runs)
        NonWeight_runs = np.array(NonWeight_runs)
        NonRtn_runs = np.array(NonRtn_runs)
        NaiveRtn_runs = np.array(NaiveRtn_runs)


        sub_folder = 'mv_Nonrobust_risk_{}'.format(risk_aver)
                    # datetime.now().strftime('%H'), datetime.now().strftime('%M'))

        log_dir = './logs/{}'.format(sub_folder)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Save params configuration
        with open('{}/params.txt'.format(log_dir), 'w') as fp:
            fp.write('Params setting \n')
            fp.write('batch size {}, seq_len {}, n_iter {} \n'.format(batch, seq_len, n_iter))
            fp.write('Risk aversion: {} \n'.format(risk_aver))


        with open('{}/sam.pickle'.format(log_dir), 'wb') as fp:
            pickle.dump(f_runs, fp)

        with open('{}/NonWeight.pickle'.format(log_dir), 'wb') as fp:
            pickle.dump(NonWeight_runs, fp)

        with open('{}/NonRtn.pickle'.format(log_dir), 'wb') as fp:
            pickle.dump(NonRtn_runs, fp)

        with open('{}/NaiveRtn.pickle'.format(log_dir), 'wb') as fp:
            pickle.dump(NaiveRtn_runs, fp)
