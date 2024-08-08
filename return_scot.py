# Worst return estimation by SCOT
import torch
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import argparse

from nets import basis_net, rtn_gen
from configs import *
from utils import *

def obj(x):
    x = torch.mean(x, dim=2)
    x = torch.prod(1 + x, dim=1)
    x = x - 1
    return -x


def c(x, y, p=1):
    '''
    L1 distance between vectors, using expanding and hence is more memory intensive
    :param x: x is tensor of shape [batch_size, time steps, features]
    :param y: y is tensor of shape [batch_size, time steps, features]
    :param p: power
    :return: cost matrix: a matrix of size [batch_size, batch_size]
    '''
    x_col = x.unsqueeze(1)
    y_lin = y.unsqueeze(0)
    b = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
    c = torch.sum(b, -1)
    return c/10.0



def train(gen_Y, f, idx, args):
    radius = args.radius
    lam_init = args.lam_init
    causal =args.causal

    test_H = basis_net(in_size=n_stock, hid_size=32, out_size=n_stock, init=False, req_grad=causal).to(device)
    test_M = basis_net(in_size=n_stock, hid_size=32, out_size=n_stock, init=False, req_grad=causal).to(device)
    var_hist = torch.zeros(n_iter, device='cpu')
    lam_hist = torch.zeros(n_iter, device='cpu')
    cost_hist = torch.zeros(n_iter, device='cpu')
    f_mean = torch.zeros(n_iter, device='cpu')


    lam = torch.tensor([lam_init], requires_grad=True, device=device)

    optimMH = optim.Adam(list(test_M.parameters()) + list(test_H.parameters()), lr=1e-3)
    optimY = optim.Adam(list(gen_Y.parameters()), lr=1e-3)

    velocity = 0.0

    x = return_sample(start_idx=idx, batch=batch, seq_len=seq_len)
    x = torch.tensor(x, dtype=torch.float, device=device)


    for iter in range(n_iter):

        ## Inner minimization over lambda, h, g
        g = test_M(x)
        y = gen_Y(x).detach()
        h = test_H(y)[:, :-1, :]*(1 + 1/(1 + 0.01*iter))

        fy = f(y).detach()
        wass_sam, pi = compute_sinkhorn(x, y, h, g, lam, fy, c)
        in_loss = lam*radius + wass_sam + martingale_regularization(g)
        test_H.zero_grad()
        test_M.zero_grad()
        lam.grad = None
        in_loss.backward()
        D_lam = lam.grad
        optimMH.step()
        velocity = 0.9*velocity - 1/(0.1*iter + 10)*D_lam
        lam = torch.clamp(lam + velocity, min=0.0).clone().detach().requires_grad_()

        with torch.no_grad():
            for param in test_M.parameters():
                param.clamp_(-50.0, 50.0)
            for param in test_H.parameters():
                param.clamp_(-50.0, 50.0)



        #### Maximization over Generator ######
        g = test_M(x).detach()
        y = gen_Y(x)
        h = test_H(y)[:, :-1, :]*(1 + 1/(1 + 0.01*iter))
        fy = f(y)

        out_wass, out_pi = compute_sinkhorn(x, y, h, g, lam.detach(), fy, c)
        out_loss = -out_wass

        gen_Y.zero_grad()
        out_loss.backward()
        optimY.step()

        # calculate H*M
        DeltaM = g[:, 1:, :] - g[:, :-1, :]
        time_steps = h.shape[1]
        sum_over_j = torch.sum(h[:, None, :, :] * DeltaM[None, :, :, :], -1)
        C_hM = torch.sum(sum_over_j, -1) / time_steps
        HMPi = torch.sum(C_hM*out_pi)

        var_hist[iter] = -out_loss.item() + lam.item()*radius
        lam_hist[iter] = lam.item()
        cost_hist[iter] = torch.sum(c(x,y)*out_pi).item()
        f_mean[iter] = -f(y).mean().item()


        # if iter % 200 == 0:
        #     # print('y', y[0, :, :])
        #     print('iter', iter, 'dual', var_hist[iter].item(), 'f(y)', f_mean[iter],
        #           'HMPi', HMPi.mean().item(), 'lam', lam.item(), 'cost', torch.sum(c(x, y) * out_pi).item())



    return var_hist.numpy(), lam_hist.numpy(), cost_hist.numpy(), f_mean.numpy()

############## Main #######################
if __name__ == '__main__':
    for radi in [0.05, 0.1, 0.2]:
        print(bcolors.GREEN + 'Current radius', radi, bcolors.ENDC)

        parser = argparse.ArgumentParser(description='Worst return estimation')
        parser.add_argument('--radius', type=float, default=radi, help='Wasserstein ball radius')
        parser.add_argument('--lam_init', type=float, default=10.0, help='Initial value of lambda')
        parser.add_argument('--cot', dest='causal', action='store_true')
        parser.add_argument('--ot', dest='causal', action='store_false')
        parser.set_defaults(causal=True)
        args = parser.parse_args()


        dual_runs = []
        lam_runs = []
        cost_runs = []
        f_runs = []

        NaiveRtn_runs = []

        for scene in range(40):

            scene = scene*5
            gen_Y = rtn_gen(in_size=n_stock, out_size=n_stock).to(device)

            # # ######### Pretraining of generator Y ##########
            # criterion = nn.MSELoss()
            # gen_opt = optim.Adam(list(gen_Y.parameters()), lr=1e-4)
            #
            # for ind in range():
            #     # in_x = torch.rand(batch, seq_len, n_stock, dtype=torch.float, device=device)
            #     # in_x = (in_x - 0.5) / 0.5*0.1
            #     # in_x[:, :, 0] += 0.1
            #     # in_x[:, :, 4] -= 0.1
            #
            #     x = return_sample(start_idx=idx, batch=batch, seq_len=seq_len)
            #     x = torch.tensor(x, dtype=torch.float, device=device)
            #
            #
            #     gen_loss = criterion(gen_Y(in_x), in_x)
            #     gen_Y.zero_grad()
            #     gen_loss.backward()
            #     gen_opt.step()
            #
            #     if ind % 50 == 0:
            #         print('gen loss', gen_loss.item())

            var_hist, lam_hist, cost_hist, f_mean = train(gen_Y, obj, scene, args)

            print('Worst return:', f_mean[-10:].mean())

            out_of_sample_idx = scene + batch + seq_len
            out_of_sample_x = return_sample(out_of_sample_idx, batch=1, seq_len=seq_len)


            naive_weights = np.ones(n_stock)/n_stock
            naive_rtn = np.matmul(out_of_sample_x, naive_weights)
            naive_rtn = np.prod(1 + naive_rtn)
            naive_rtn = naive_rtn - 1
            print('Naive return:', naive_rtn)

            NaiveRtn_runs.append(naive_rtn)


            dual_runs.append(var_hist)
            lam_runs.append(lam_hist)
            cost_runs.append(cost_hist)
            f_runs.append(f_mean)


        dual_runs = np.array(dual_runs)
        lam_runs = np.array(lam_runs)
        cost_runs = np.array(cost_runs)
        f_runs = np.array(f_runs)

        NaiveRtn_runs = np.array(NaiveRtn_runs)

        sub_folder = 'RtnEst_SCOT_radi_{}'.format(args.radius)
        log_dir = './logs/{}'.format(sub_folder)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Save params configuration
        with open('{}/params.txt'.format(log_dir), 'w') as fp:
            fp.write('Params setting \n')
            fp.write('batch size {}, seq_len {}, n_iter {} \n'.format(batch, seq_len, n_iter))
            fp.write('radius: {} \n'.format(args.radius))
            fp.write('lambda init: {} \n'.format(args.lam_init))

        with open('{}/dual.pickle'.format(log_dir), 'wb') as fp:
            pickle.dump(dual_runs, fp)

        with open('{}/sam.pickle'.format(log_dir), 'wb') as fp:
            pickle.dump(f_runs, fp)

        with open('{}/lam.pickle'.format(log_dir), 'wb') as fp:
            pickle.dump(lam_runs, fp)

        with open('{}/cost.pickle'.format(log_dir), 'wb') as fp:
            pickle.dump(cost_runs, fp)

        with open('{}/NaiveRtn.pickle'.format(log_dir), 'wb') as fp:
            pickle.dump(NaiveRtn_runs, fp)



