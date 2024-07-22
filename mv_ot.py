# Real parameter given
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import pickle
import argparse
import os
from datetime import datetime
from vol_config import *
from nets import mv_obj, vol_obj





def cost(x, y):
    return (x - y).abs().pow(1).sum(axis=(1, 2))/x.shape[0]

def beta(x, lam, f):
    y = (x + torch.rand_like(x) * 1e-3).clone().requires_grad_()
    t = 0
    b_old = 10
    b = f(y).sum() - lam*cost(x, y).sum()

    while t<40 and torch.abs((b - b_old)/b_old) > 1e-3:
        b_old = b.item()
        t += 1
        b.backward(retain_graph=True)
        y = torch.clamp(y + 1/(t+1)*y.grad, min=-0.5, max=0.5).clone().detach().requires_grad_()
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
    # x = torch.rand(batch, seq_len, n_stock, dtype=torch.float, device=device)
    # x = (x - 0.5) / 0.5 * 0.1
    # x[:, :, 0] += 0.1
    # x[:, :, 4] -= 0.1
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

            # if iter % 100 == 0:
            #     for name, param in f.named_parameters():
            #         print(name, param.data)
            #     print('Theoretical return', (1 + torch.dot(f.weight.data, torch.tensor([0.1, 0.0, 0.0, 0.0, -0.1],
            #                                                                            device=device))) ** 5 - 1)

            # for param in f.parameters():
            #     param.clamp_(0.0, 1.0)

        dual_hist[iter] = dual.item()
        lam_hist[iter] = lam.item()
        cost_hist[iter] = cost(x, y).mean().item()
        weights[iter, :] = f.weight.data.cpu()

        sam_mean[iter] = f(y).mean().item()
        if iter % 200 == 0:
            # print(y[0, :, :])
            print('iter', iter, 'dual', dual.item(), 'lam', lam.item(),
                  'f mean', f(y).mean().item(), 'cost mean', cost(x, y).mean().item())

    # x_last = x[-1, :, :].reshape(-1).cpu().numpy()
    # y_last = y[-1, :, :].reshape(-1).detach().cpu().numpy()

    return dual_hist.numpy(), lam_hist.numpy(), sam_mean.numpy(), cost_hist.numpy(), weights.numpy()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Volatility estimation')
    parser.add_argument('--radius', type=float, default=0.1, help='Wasserstein ball radius')
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

    risk_aver = 1.0

    for scene in range(n_runs):
        scene = scene*10

        f = mv_obj(weight_init=torch.ones(n_stock)/n_stock,
                   mean_init=torch.ones(1), risk_aversion=risk_aver).to(device)
        dual_hist, lam_hist, sam_mean, cost_hist, weights = train(f, scene, args)

        out_of_sample_idx = scene + batch + seq_len
        # weights = f.weight.data.cpu().numpy()
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
        # x_hist.append(x_last)
        # y_hist.append(y_last)
        cost_runs.append(cost_hist)

    dual_runs = np.array(dual_runs)
    lam_runs = np.array(lam_runs)
    sam_runs = np.array(sam_runs)
    cost_runs = np.array(cost_runs)
    # x_hist = np.array(x_hist)
    # y_hist = np.array(y_hist)

    DROWeight_runs = np.array(DROWeight_runs)
    DRORtn_runs = np.array(DRORtn_runs)
    NaiveRtn_runs = np.array(NaiveRtn_runs)


    sub_folder = 'mv_ot_rad_{}_aver_{}'.format(args.radius, risk_aver) #datetime.now().strftime('%H'), datetime.now().strftime('%M'))
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

    # with open('{}/x_hist.pickle'.format(log_dir), 'wb') as fp:
    #     pickle.dump(x_hist, fp)
    #
    # with open('{}/y_hist.pickle'.format(log_dir), 'wb') as fp:
    #     pickle.dump(y_hist, fp)

    with open('{}/DROWeight.pickle'.format(log_dir), 'wb') as fp:
        pickle.dump(DROWeight_runs, fp)

    with open('{}/DRORtn.pickle'.format(log_dir), 'wb') as fp:
        pickle.dump(DRORtn_runs, fp)

    with open('{}/NaiveRtn.pickle'.format(log_dir), 'wb') as fp:
        pickle.dump(NaiveRtn_runs, fp)
