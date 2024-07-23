# Real parameter given
import torch
import torch.optim as optim
import pickle
import argparse
import os
from configs import *


def obj(x):
    x = torch.mean(x, dim=2)
    x = torch.prod(1 + x, dim=1)
    x = x - 1
    return -x


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



    lam = torch.tensor([lam_init], requires_grad=True, device=device)
    velocity = 0.0

    # load return data
    x = return_sample(start_idx=idx, batch=batch, seq_len=seq_len)
    x = torch.tensor(x, dtype=torch.float, device=device)

    for iter in range(n_iter):

        y = beta(x, lam.detach(), f)

        b = f(y.detach()).detach().sum() - lam * cost(x, y.detach()).sum()
        dual = lam * radius + b/batch
        obj = dual

        lam.grad = None
        obj.backward()
        D_lam = lam.grad

        velocity = 0.9*velocity - 1/(0.1*iter+10)*D_lam
        lam = torch.clamp(lam + velocity, min=0.0).clone().detach().requires_grad_()


        dual_hist[iter] = dual.item()
        lam_hist[iter] = lam.item()
        cost_hist[iter] = cost(x, y).mean().item()

        sam_mean[iter] = -f(y).mean().item()
        # if iter % 200 == 0:
        #     print(y[0, :, :])
        #     print('iter', iter, 'dual', dual.item(), 'lam', lam.item(),
        #           'worst return', sam_mean[iter], 'cost mean', cost(x, y).mean().item())


    return dual_hist.numpy(), lam_hist.numpy(), sam_mean.numpy(), cost_hist.numpy()

if __name__ == '__main__':
    for radi in [0.1]:
        print(bcolors.GREEN + 'Current radius', radi, bcolors.ENDC)

        parser = argparse.ArgumentParser(description='Volatility estimation')
        parser.add_argument('--radius', type=float, default=radi, help='Wasserstein ball radius')
        parser.add_argument('--lam_init', type=float, default=10.0, help='Initial value of lambda')
        args = parser.parse_args()

        dual_runs = []
        lam_runs = []
        sam_runs = []
        cost_runs = []

        NaiveRtn_runs = []

        for scene in range(9):
            scene = scene*10


            dual_hist, lam_hist, sam_mean, cost_hist = train(obj, scene, args)
            print('worst return', sam_mean[-10:].mean())

            out_of_sample_idx = scene + batch + seq_len
            out_of_sample_x = return_sample(out_of_sample_idx, batch=1, seq_len=seq_len)

            naive_weights = np.ones(n_stock) / n_stock
            naive_rtn = np.matmul(out_of_sample_x, naive_weights)
            naive_rtn = np.prod(1 + naive_rtn)
            naive_rtn = naive_rtn - 1
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


        NaiveRtn_runs = np.array(NaiveRtn_runs)

        sub_folder = 'RtnEst_OT_radi_{}'.format(args.radius)
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
            pickle.dump(sam_runs, fp)

        with open('{}/lam.pickle'.format(log_dir), 'wb') as fp:
            pickle.dump(lam_runs, fp)

        with open('{}/cost.pickle'.format(log_dir), 'wb') as fp:
            pickle.dump(cost_runs, fp)

        with open('{}/NaiveRtn.pickle'.format(log_dir), 'wb') as fp:
            pickle.dump(NaiveRtn_runs, fp)

