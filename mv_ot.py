# OT method for mean-variance portfolio
import torch
import numpy as np
import torch.optim as optim
import pickle
import argparse
import os
from datetime import datetime
from configs import *
from nets import mv_obj



def cost(x, y):
    return (x - y).abs().pow(1).sum(axis=(1, 2))/10.0

def beta(x, lam, f):
    y = (x + torch.rand_like(x) * 1e-3).clone().requires_grad_()
    t = 0
    b_old = 10
    b = f(y).sum() - lam*cost(x, y).sum()

    while t<40 and torch.abs((b - b_old)/b_old) > 1e-3:
        b_old = b.item()
        t += 1
        b.backward(retain_graph=True)
        y = torch.clamp(y + 1/(t+1)*y.grad, min=-0.2, max=0.2).clone().detach().requires_grad_()
        b = f(y).sum() - lam * cost(x, y).sum()
    return y


####################### Training ##################################

def train(f, idx, args):
    radius = args.radius
    lam_init = args.lam_init
    dual_hist = torch.zeros(n_iter, device='cpu')
    lam_hist = torch.zeros(n_iter, device='cpu')
    cost_hist = torch.zeros(n_iter, device='cpu')
    sam_mean = torch.zeros(n_iter, device='cpu')
    weights = torch.zeros((n_iter, n_stock), device='cpu')


    lam = torch.tensor([lam_init], requires_grad=True, device=device)
    velocity = 0.0

    optimF = optim.Adam(list(f.parameters()), lr=1e-2)

    # load return data
    x = return_sample(start_idx=idx, batch=batch, seq_len=seq_len)
    x = torch.tensor(x, dtype=torch.float, device=device)

    for iter in range(n_iter):

        y = beta(x, lam.detach(), f)

        b = f(y.detach()).detach().sum() - lam * cost(x, y.detach()).sum()
        dual = lam * radius + b/batch # empirical measure, divided by 1/batch_size
        obj = dual

        lam.grad = None
        obj.backward()
        D_lam = lam.grad

        velocity = 0.9*velocity - 1/(0.1*iter+10)*D_lam
        lam = torch.clamp(lam + velocity, min=0.0).clone().detach().requires_grad_()

        ## Minimize over investment weights and target
        y = beta(x, lam.detach(), f)
        b = f(y.detach()).sum() - lam.detach() * cost(x, y.detach()).sum()
        out_loss = b/batch
        f.zero_grad()
        out_loss.backward()
        optimF.step()

        with torch.no_grad():
            f.weight.clamp_(1e-5, 1.0)
            # self.transition.data.div_(torch.sum(model.transition.data, dim=0))
            # f.weight.div_(torch.sum(f.weight.data)) #= f.weight.data/f.weight.data.sum()
            f.weight += (1 - f.weight.data.sum()) / n_stock


        dual_hist[iter] = dual.item()
        lam_hist[iter] = lam.item()
        cost_hist[iter] = cost(x, y).mean().item()
        weights[iter, :] = f.weight.data.cpu()

        sam_mean[iter] = f(y).mean().item()
        # if iter % 1000 == 0:
        #     print(y[0, :, :])
        #     print(x[0, :, :])
        #     print((x-y).detach().abs().sum()/10)
        #     print('iter', iter, 'dual', dual.item(), 'lam', lam.item(),
        #           'f mean', f(y).mean().item(), 'cost mean', cost(x, y).mean().item())



    return dual_hist.numpy(), lam_hist.numpy(), sam_mean.numpy(), cost_hist.numpy(), weights.numpy()

if __name__ == '__main__':
    for risk_aver in [0.0, 0.1, 1.0]:
        for radi in [0.05, 0.1, 0.2]:

            print(bcolors.GREEN + 'Current risk aversion:', risk_aver, 'radius', radi, bcolors.ENDC)

            parser = argparse.ArgumentParser(description='Mean-variance portfolios with OT')
            parser.add_argument('--radius', type=float, default=radi, help='Wasserstein ball radius')
            parser.add_argument('--lam_init', type=float, default=10.0, help='Initial value of lambda')
            args = parser.parse_args()

            dual_runs = []
            lam_runs = []
            sam_runs = []
            cost_runs = []
            x_hist = []
            y_hist = []

            DRORtn_runs = []
            DROWeight_runs = []
            NaiveRtn_runs = []


            for scene in range(n_runs):
                scene = scene*10

                f = mv_obj(weight_init=torch.ones(n_stock)/n_stock,
                           mean_init=torch.ones(1), risk_aversion=risk_aver).to(device)
                dual_hist, lam_hist, sam_mean, cost_hist, weights = train(f, scene, args)

                out_of_sample_idx = scene + batch + seq_len
                last_weights = weights[-10:, :].mean(axis=0)
                out_of_sample_x = return_sample(out_of_sample_idx, batch=1, seq_len=seq_len)
                out_of_sample_rtn = np.matmul(out_of_sample_x, last_weights)
                out_of_sample_rtn = np.prod(1 + out_of_sample_rtn)
                print('DRO weights:', last_weights)
                print('DRO return:', out_of_sample_rtn)

                DROWeight_runs.append(last_weights)
                DRORtn_runs.append(out_of_sample_rtn)


                naive_weights = np.ones(n_stock) / n_stock
                naive_rtn = np.matmul(out_of_sample_x, naive_weights)
                naive_rtn = np.prod(1 + naive_rtn)
                print('Naive return:', naive_rtn)

                NaiveRtn_runs.append(naive_rtn)


                dual_runs.append(dual_hist)
                lam_runs.append(lam_hist)
                sam_runs.append(sam_mean)
                cost_runs.append(cost_hist)

            dual_runs = np.array(dual_runs)
            lam_runs = np.array(lam_runs)
            sam_runs = np.array(sam_runs)
            cost_runs = np.array(cost_runs)


            DROWeight_runs = np.array(DROWeight_runs)
            DRORtn_runs = np.array(DRORtn_runs)
            NaiveRtn_runs = np.array(NaiveRtn_runs)


            sub_folder = 'mv_OT_radi_{}_risk_{}'.format(args.radius, risk_aver)
            log_dir = './logs/{}'.format(sub_folder)

            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            # Save params configuration
            with open('{}/params.txt'.format(log_dir), 'w') as fp:
                fp.write('Params setting \n')
                fp.write('batch size {}, seq_len {}, n_iter {} \n'.format(batch, seq_len, n_iter))
                fp.write('radius: {} \n'.format(args.radius))
                fp.write('risk aversion: {} \n'.format(risk_aver))
                fp.write('lambda init: {} \n'.format(args.lam_init))


            with open('{}/dual.pickle'.format(log_dir), 'wb') as fp:
                pickle.dump(dual_runs, fp)

            with open('{}/sam.pickle'.format(log_dir), 'wb') as fp:
                pickle.dump(sam_runs, fp)

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
