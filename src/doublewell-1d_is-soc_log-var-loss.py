import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

from base_parser import get_base_parser
from models import FeedForwardNN

def get_parser():
    parser = get_base_parser()
    return parser

def gradient(x, alpha):
    return 4 * alpha * x * (x**2 - 1)

def f(x):
    return torch.tensor([1.], dtype=torch.float32)

def g(x):
    return torch.tensor([0.], dtype=torch.float32)

def get_idx_new_in_ts(x, been_in_target_set):
    is_in_target_set = x > 1

    idx = torch.where(
            (is_in_target_set == True) &
            (been_in_target_set == False)
    )[0]

    been_in_target_set[idx] = True

    return idx

def sample_loss_vectorized(model, dt, K):
    alpha = torch.tensor([1.], dtype=torch.float32)
    beta = torch.tensor(1., dtype=torch.float32)
    sigma = torch.sqrt(2 / beta)
    dt = torch.tensor(dt, dtype=torch.float32)
    k_max = 10**6

    # start timer
    ct_initial = time.time()

    # initialize trajectories of processes X and Y 
    x_init = torch.tensor([-1.], dtype=torch.float32)
    xt = x_init.repeat((K, 1))
    yt = torch.zeros(K)
    y_fht = torch.empty(K)

    # initialize work and running integral
    work_t = torch.zeros(K)
    work_fht = torch.empty(K)
    det_int_t = torch.zeros(K)
    det_int_fht = torch.empty(K)
    stoch_int_t = torch.zeros(K)
    stoch_int_fht = torch.empty(K)

    # preallocate time steps
    time_steps = np.empty(K)

    been_in_target_set = torch.full((K, 1), False)

    for k in np.arange(1, k_max + 1):

        # Brownian increment
        dbt = torch.sqrt(dt) * torch.normal(torch.tensor(0.), torch.tensor(1.), size=(K, 1))

        # control
        zt = - model.forward(xt)

        # adaptive forward process
        ut = - zt.detach()

        # update work with running cost
        work_t = work_t + f(xt) * dt

        # update deterministic integral
        zt_normed_squared = torch.linalg.norm(zt, axis=1) ** 2
        det_int_t = det_int_t + zt_normed_squared * dt

        # update stochastic integral
        zt_dbt_product = torch.matmul(
            zt[:, np.newaxis, :],
            dbt[:, :, np.newaxis],
        ).squeeze()
        stoch_int_t = stoch_int_t - zt_dbt_product

        # sde update process xt
        xt = xt + (- gradient(xt, alpha) + sigma * ut) * dt + sigma * dbt

        # ut and zt product
        ut_zt_product = torch.matmul(
            ut[:, np.newaxis, :],
            zt[:, :, np.newaxis],
        ).squeeze()

        # sde update process yt
        yt = yt + (-f(xt) + 0.5 * zt_normed_squared + ut_zt_product) * dt \
           + zt_dbt_product

        # get indices of trajectories which are new to the target set
        idx = get_idx_new_in_ts(xt, been_in_target_set)

        if idx.shape[0] != 0:

            # update work and yt process with the final cost
            work_t = work_t + g(xt)
            yt = yt + g(xt)

            # fix work
            work_fht[idx] = work_t.index_select(0, idx)

            # fix yt process
            y_fht[idx] = yt.index_select(0, idx)

            # fix running integrals
            det_int_fht[idx] = det_int_t.index_select(0, idx)
            stoch_int_fht[idx] = stoch_int_t.index_select(0, idx)

            # time steps
            time_steps[idx] = k

        # stop if xt_traj in target set
        if been_in_target_set.all() == True:
           break

    # compute cost functional (loss)
    phi_fht = (work_fht + 0.5 * det_int_fht).detach().numpy()

    # compute log variance loss
    log_var_loss = torch.var(y_fht)

    # compute mean and re of I_u
    I_u = np.exp(
        - work_fht.numpy()
        - stoch_int_fht.detach().numpy()
        - 0.5 * det_int_fht.detach().numpy()
    )
    mean_I_u = np.mean(I_u)
    var_I_u = np.var(I_u)
    re_I_u = np.sqrt(var_I_u) / mean_I_u

    # end timer
    ct_final = time.time()

    return log_var_loss, phi_fht, mean_I_u, var_I_u, re_I_u, time_steps, ct_final - ct_initial

def sample_loss_vectorized_stopped(model, dt, K):
    alpha = torch.tensor([1.], dtype=torch.float32)
    beta = torch.tensor(1., dtype=torch.float32)
    sigma = torch.sqrt(2 / beta)
    dt = torch.tensor(dt, dtype=torch.float32)
    k_max = 10**6

    # start timer
    ct_initial = time.time()

    # initialize trajectories of processes X and Y 
    x_init = torch.tensor([-1.], dtype=torch.float32)
    xt = x_init.repeat((K, 1))
    yt = torch.zeros(K)

    # initialize fht
    work_t = torch.zeros(K)

    # initialize running integrals
    det_int_t = torch.zeros(K)
    stoch_int_t = torch.zeros(K)

    # preallocate time steps
    time_steps = np.empty(K)

    # number of trajectories not in the target set
    K_not_in_ts = K

    for k in np.arange(1, k_max + 1):

        # stop trajetories if all trajectories are in the target set
        idx = xt[:, 0] < 1
        K_not_in_ts = torch.sum(idx)
        if K_not_in_ts == 0:
            break

        # Brownian increment
        dbt = torch.sqrt(dt) * torch.randn(K_not_in_ts).reshape(K_not_in_ts, 1)

        # Z process
        zt = - model.forward(xt[idx, :])

        # adaptive forward process
        ut = - zt.detach()

        # update work with running cost
        work_t[idx] = work_t[idx] + f(xt[idx]) * dt

        # update deterministic integral
        zt_normed_squared = torch.linalg.norm(zt, axis=1) ** 2
        det_int_t[idx] = det_int_t[idx] + zt_normed_squared * dt

        # update stochastic integral
        zt_dbt_product = torch.matmul(
            zt[:, np.newaxis, :],
            dbt[:, :, np.newaxis],
        ).squeeze()
        stoch_int_t[idx] = stoch_int_t[idx] - zt_dbt_product

        # sde update process xt
        xt[idx, :] = xt[idx, :] \
                   + (- gradient(xt[idx, :], alpha) + sigma * ut) * dt \
                   + sigma * dbt

        # ut and zt product
        ut_zt_product = torch.matmul(
            ut[:, np.newaxis, :],
            zt[:, :, np.newaxis],
        ).squeeze()

        # sde update process xt
        yt[idx] = yt[idx] \
                + (- f(xt[idx]) + 0.5 * zt_normed_squared + ut_zt_product) * dt \
                + zt_dbt_product

        # numpy array indices
        idx = idx.detach().numpy()

        # time steps
        time_steps[idx] += 1


    # compute loss
    log_var_loss = torch.var(yt)

    # compute cost functional (loss)
    phi_t = (work_t + 0.5 * det_int_t).detach().numpy()

    # compute mean and re of I_u
    I_u = np.exp(
        - work_t.numpy()
        - stoch_int_t.detach().numpy()
        - 0.5 * det_int_t.detach().numpy()
    )
    mean_I_u = np.mean(I_u)
    var_I_u = np.var(I_u)
    re_I_u = np.sqrt(var_I_u) / mean_I_u

    # end timer
    ct_final = time.time()

    return log_var_loss, phi_t, mean_I_u, var_I_u, re_I_u, time_steps, ct_final - ct_initial


def plot_controls(domain_h, controls):
    n_controls = controls.shape[0]

    fig, ax = plt.subplots()
    for i in range(n_controls):
        if i % 10 == 0:
            ax.plot(domain_h, controls[i])
    ax.set_ylim(-3, 3)
    plt.show()

def plot_losses(losses):
    fig, ax = plt.subplots()
    ax.plot(losses)
    plt.show()

def main():
    args = get_parser().parse_args()

    # fix seed
    if args.seed is not None:
        np.random.seed(args.seed)

    # get dimensions of each layer
    d_hidden_layers = [args.d_hidden_layer for i in range(args.n_layers-1)]

    # initialize nn model 
    model = FeedForwardNN(d_in=args.d, hidden_sizes=d_hidden_layers, d_out=args.d)

    # define optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    # discretized domain
    h = 0.01
    domain_h = torch.arange(-2, 2+h, h).unsqueeze(dim=1)
    Nh = domain_h.shape[0]

    # preallocate parameters
    losses = np.empty(args.n_iterations)
    var_losses = np.empty(args.n_iterations)
    controls = np.empty((args.n_iterations, Nh))
    means_I_u = np.empty(args.n_iterations)
    vars_I_u = np.empty(args.n_iterations)
    res_I_u = np.empty(args.n_iterations)
    #avg_time_steps = np.empty(args.n_iterations, dtype=np.int64)
    avgs_time_steps = np.empty(args.n_iterations)
    cts = np.empty(args.n_iterations)

    # save initial control
    controls[0] = model.forward(domain_h).detach().numpy().squeeze()

    for i in np.arange(args.n_iterations):

        # reset gradients
        optimizer.zero_grad()

        # compute effective loss and relative entropy loss (phi_fht)
        log_var_loss, phi_fht, mean_I_u, var_I_u, re_I_u, time_steps, ct = sample_loss_vectorized(model, args.dt, args.K)
        #log_var_loss, phi_fht, mean_I_u, var_I_u, re_I_u, time_steps, ct = sample_loss_vectorized_stopped(model, args.dt, args.K)
        log_var_loss.backward()

        # update parameters
        optimizer.step()

        # compute loss and variance
        loss = np.mean(phi_fht)
        var = np.var(phi_fht)

        # average time steps
        avg_time_steps = np.mean(time_steps)

        msg = 'it.: {:2d}, loss: {:.3e}, var: {:.1e}, mean I^u: {:.3e}, var I^u: {:.1e}, ' \
              're I^u: {:.1e}, avg ts: {:.3e}, ct: {:.3f}' \
              ''.format(i, loss, var, mean_I_u, var_I_u, re_I_u, avg_time_steps, ct)
        print(msg)

        # save statistics
        losses[i] = loss
        var_losses[i] = var
        means_I_u[i] = mean_I_u
        vars_I_u[i] = var_I_u
        res_I_u[i] = re_I_u
        avgs_time_steps[i] = avg_time_steps
        cts[i] = ct

        # save control
        controls[i] = model.forward(domain_h).detach().numpy().squeeze()

    # plot control
    plot_controls(domain_h, controls)
    plot_losses(losses)

if __name__ == "__main__":
    main()

