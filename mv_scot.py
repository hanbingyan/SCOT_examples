# Mean-variance portfolio selection
import torch
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import argparse
from datetime import datetime
from nets import basis_net, rtn_gen, mv_obj
from configs import *
from utils import *




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
    b = torch.sum((torch.abs(x_col - y_lin))**p, -1)
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
    weights = torch.zeros((n_iter, n_stock), device='cpu')

    lam = torch.tensor([lam_init], requires_grad=True, device=device)

    optimMH = optim.Adam(list(test_M.parameters()) + list(test_H.parameters()), lr=1e-3)
    optimY = optim.Adam(list(gen_Y.parameters()), lr=1e-3)
    optimF = optim.Adam(list(f.parameters()), lr=1e-2)
    velocity = 0.0

    # load return data
    # with open('x52.pickle', 'rb') as fp:
    #     x_tar = pickle.load(fp)
    # x = torch.tensor(x_tar, dtype=torch.float, device=device)

    # x = torch.rand(batch, seq_len, n_stock, dtype=torch.float, device=device)
    # x = (x - 0.5)/0.5*0.1
    # x[:, :, 0] += 0.1
    # x[:, :, 4] -= 0.1

    x = return_sample(start_idx=idx, batch=batch, seq_len=seq_len)
    x = torch.tensor(x, dtype=torch.float, device=device)


    for iter in range(n_iter):

        ## Inner minimization over lambda, h, g
        g = test_M(x)
        y = gen_Y(x).detach()
        h = test_H(y)[:, :-1, :]*(1 + 1/(1 + 0.01*iter))

        fy = f(y).detach()
        wass_sam, pi = compute_sinkhorn(x, y, h, g, lam, fy, c)
        in_loss = lam * radius + wass_sam + martingale_regularization(g)
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
        fy = f(y) # no detach here

        out_wass, out_pi = compute_sinkhorn(x, y, h, g, lam.detach(), fy, c)
        out_loss = -out_wass

        gen_Y.zero_grad()
        out_loss.backward()
        optimY.step()

        ## Minimize over investment weights and target
        g = test_M(x).detach()
        y = gen_Y(x).detach()
        h = test_H(y)[:, :-1, :]*(1 + 1/(1 + 0.01*iter))
        h = h.detach()
        fy = f(y)
        wass_sam, pi = compute_sinkhorn(x, y, h, g, lam.detach(), fy, c)
        in_loss = wass_sam
        f.zero_grad()
        in_loss.backward()
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

        # calculate H*M
        DeltaM = g[:, 1:, :] - g[:, :-1, :]
        time_steps = h.shape[1]
        sum_over_j = torch.sum(h[:, None, :, :] * DeltaM[None, :, :, :], -1)
        C_hM = torch.sum(sum_over_j, -1) / time_steps
        HMPi = torch.sum(C_hM*out_pi)

        var_hist[iter] = -out_loss.item() + lam.item()*radius  # not really used in later analysis
        lam_hist[iter] = lam.item()
        cost_hist[iter] = torch.sum(c(x,y)*out_pi).item()
        f_mean[iter] = f(y).mean().item()
        weights[iter, :] = f.weight.data.cpu()

        # if iter % 1000 == 0:
        #     print(y[0, :, :])
        #     print(x[0, :, :].detach().mean())
        #     print((x-y).detach().abs().sum()/10)
        #     print('iter', iter, 'dual', var_hist[iter].item(), 'f(y)', f(y).mean().item(),
        #           'HMPi', HMPi.mean().item(), 'lam', lam.item(), 'cost', torch.sum(c(x, y) * out_pi).item())

    return var_hist.numpy(), lam_hist.numpy(), cost_hist.numpy(), f_mean.numpy(), weights.numpy()

############## Main #######################
if __name__ == '__main__':

    for risk_aver in [0.0, 0.1, 1.0]:
        for radi in [0.05, 0.1, 0.2]:
            print(bcolors.GREEN + 'Current risk aversion:', risk_aver, 'radius', radi, bcolors.ENDC)

            parser = argparse.ArgumentParser(description='Mean-variance portfolios with SCOT')
            parser.add_argument('--radius', type=float, default=radi, help='Wasserstein ball radius')
            parser.add_argument('--lam_init', type=float, default=10.0, help='Initial value of lambda')
            parser.add_argument('--cot', dest='causal', action='store_true')
            parser.add_argument('--ot', dest='causal', action='store_false')
            parser.set_defaults(causal=True)   # Set True for causal OT
            args = parser.parse_args()

            dual_runs = []
            lam_runs = []
            cost_runs = []
            f_runs = []

            DRORtn_runs = []
            DROWeight_runs = []
            NaiveRtn_runs = []

            for scene in range(n_runs):

                scene = scene*10

                f = mv_obj(weight_init=torch.ones(n_stock)/n_stock, mean_init=torch.ones(1),
                           risk_aversion=risk_aver).to(device)
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

                var_hist, lam_hist, cost_hist, f_mean, weights = train(gen_Y, f, scene, args)


                out_of_sample_idx = scene + batch + seq_len
                # To increase stability, use the average of the last 10 weights
                last_weights = weights[-10:, :].mean(axis=0)
                out_of_sample_x = return_sample(out_of_sample_idx, batch=1, seq_len=seq_len)
                out_of_sample_rtn = np.matmul(out_of_sample_x, last_weights)
                out_of_sample_rtn = np.prod(1 + out_of_sample_rtn)
                print('DRO weights:', last_weights)
                print('DRO return (+1.0):', out_of_sample_rtn)

                DROWeight_runs.append(last_weights)
                DRORtn_runs.append(out_of_sample_rtn)

                naive_weights = np.ones(n_stock)/n_stock
                naive_rtn = np.matmul(out_of_sample_x, naive_weights)
                naive_rtn = np.prod(1 + naive_rtn)
                print('Naive return (+1.0):', naive_rtn)

                NaiveRtn_runs.append(naive_rtn)


                dual_runs.append(var_hist)
                lam_runs.append(lam_hist)
                cost_runs.append(cost_hist)
                f_runs.append(f_mean)


            dual_runs = np.array(dual_runs)
            lam_runs = np.array(lam_runs)
            cost_runs = np.array(cost_runs)
            f_runs = np.array(f_runs)


            DROWeight_runs = np.array(DROWeight_runs)
            DRORtn_runs = np.array(DRORtn_runs)
            NaiveRtn_runs = np.array(NaiveRtn_runs)


            sub_folder = 'mv_SCOT_radi_{}_risk_{}'.format(radi, risk_aver)
                        # datetime.now().strftime('%H'), datetime.now().strftime('%M'))

            log_dir = './logs/{}'.format(sub_folder)

            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            # Save params configuration
            with open('{}/params.txt'.format(log_dir), 'w') as fp:
                fp.write('Params setting \n')
                fp.write('COT or not: {} \n'.format(args.causal))
                fp.write('batch size {}, seq_len {}, n_iter {} \n'.format(batch, seq_len, n_iter))
                fp.write('Radius: {} \n'.format(args.radius))
                fp.write('Risk aversion: {} \n'.format(risk_aver))
                fp.write('Lambda Init: {} \n'.format(args.lam_init))

            with open('{}/dual.pickle'.format(log_dir), 'wb') as fp:
                pickle.dump(dual_runs, fp)

            with open('{}/sam.pickle'.format(log_dir), 'wb') as fp:
                pickle.dump(f_runs, fp)

            with open('{}/lam.pickle'.format(log_dir), 'wb') as fp:
                pickle.dump(lam_runs, fp)

            with open('{}/cost.pickle'.format(log_dir), 'wb') as fp:
                pickle.dump(cost_runs, fp)

            with open('{}/DROWeight.pickle'.format(log_dir), 'wb') as fp:
                pickle.dump(DROWeight_runs, fp)

            with open('{}/DRORtn.pickle'.format(log_dir), 'wb') as fp:
                pickle.dump(DRORtn_runs, fp)

            with open('{}/NaiveRtn.pickle'.format(log_dir), 'wb') as fp:
                pickle.dump(NaiveRtn_runs, fp)
