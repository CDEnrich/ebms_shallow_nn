import argparse
import numpy as np
import os, sys
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import time

from mala import get_samples, get_samples_uniform_proposal
from util import get_teacher, teacher_samples, to_bool

TEACHER_BURNIN = 10000
TEACHER_BURN = 4

def energy(x, Win, wout):
    neurons = Win.shape[1]
    return torch.sum(nn.functional.relu(torch.matmul(x, Win)) * wout, dim=1)/neurons

def loss_gradient_weights(Win, wout, X, Xmcmc):
    n, nmcmc = X.shape[0], Xmcmc.shape[0]
    neurons = Win.shape[1]
    gd_data = torch.sum(nn.functional.relu(torch.matmul(X, Win)), dim = 0) / n / neurons
    gd_model = torch.sum(nn.functional.relu(torch.matmul(Xmcmc, Win)), dim = 0) / nmcmc /neurons
    return gd_data - gd_model, gd_model

def loss_gradient_positions(Win, wout, X, Xmcmc):
    n, nmcmc = X.shape[0], Xmcmc.shape[0]
    neurons = Win.shape[1]
    gd_data = X.t().matmul(torch.sign(torch.matmul(X, Win))*0.5 + 0.5) * wout / n / neurons
    gd_model = Xmcmc.t().matmul(torch.sign(torch.matmul(Xmcmc, Win))*0.5 + 0.5) * wout / nmcmc / neurons

    return gd_data - gd_model, gd_model

def augmented_energy(X, Win, wout):
    neurons = Win.shape[0]
    return torch.sum(torch.nn.functional.relu(torch.matmul(X,Win))*wout)/(X.shape[1]*neurons)

def augmented_exp_minus_energy(X, Win, wout, compute_mean=True):
    neurons = Win.shape[0]
    values = torch.exp(-torch.sum(torch.nn.functional.relu(torch.matmul(X,Win))*wout, dim = 0)/neurons)
    if compute_mean:
        return torch.mean(values, dim = 0), torch.std(values, dim=0)
    else:
        return torch.sum(values, dim = 0)

def unnormalized_densities(X, Win, wout):
    neurons = Win.shape[0]
    return torch.exp(-torch.sum(torch.nn.functional.relu(torch.matmul(X,Win))*wout, dim = 0)/neurons)

def energies(X, Win, wout):
    neurons = Win.shape[0]
    return torch.sum(torch.nn.functional.relu(torch.matmul(X,Win))*wout, dim = 0)/neurons

def cross_entropy_avgterm(Win, wout, X):
    return float(augmented_energy(X.unsqueeze(0), torch.t(Win).unsqueeze(2), torch.t(wout).unsqueeze(2)))

def free_energy_sampling(Win, wout, n_uniform_samples=10000, reps=10, get_stuff=False):
    dimension = Win.shape[0]
    fs = []
    partitions = []
    logterms = []
    for _ in range(reps):
        uniform_samples = torch.randn(n_uniform_samples,dimension)
        uniform_samples = torch.nn.functional.normalize(uniform_samples, p=2, dim=1)
        with torch.no_grad():
            part_mean, part_std = augmented_exp_minus_energy(uniform_samples.unsqueeze(0), torch.t(Win).unsqueeze(2), torch.t(wout).unsqueeze(2), compute_mean=True)
            partitions.append((float(part_mean), float(part_std)))
            log_term = torch.log(part_mean)
            logterms.append(float(log_term))
        fs.append(log_term.detach().numpy()[0])
    f = np.mean(fs)
    ps = np.array(partitions)
    print('{} +- {}'.format(f, np.std(fs)))
    lbs = np.log(ps[:,0] - ps[:,1] / np.sqrt(n_uniform_samples))
    ubs = np.log(ps[:,0] + ps[:,1] / np.sqrt(n_uniform_samples))
    print('log Z bounds, average {}'.format((ubs - lbs).mean()))
    print('F (=log Z) mean:', np.mean(logterms))
    if get_stuff:
        return np.mean(fs), (fs, ps)
    else:
        return np.mean(fs)

def cross_entropy(Win, wout, X, n_uniform_samples=10000, reps=10):
    avg_term = cross_entropy_avgterm(Win, wout, X)
    f_term = free_energy_sampling(Win, wout, n_uniform_samples=n_uniform_samples, reps=reps)
    return avg_term + f_term

def compute_kl_a_posteriori(teacher_Win, teacher_wout, Win, wout, n_uniform_samples=200000, reps=1):
    dimension = Win.shape[0]
    kl_reps = []
    for _ in range(reps):
        uniform_samples = torch.randn(n_uniform_samples,dimension)
        uniform_samples = torch.nn.functional.normalize(uniform_samples, p=2, dim=1)
        with torch.no_grad():
            part_mean_student = augmented_exp_minus_energy(uniform_samples.unsqueeze(0), torch.t(Win).unsqueeze(2), torch.t(wout).unsqueeze(2), compute_mean=False)
            part_mean_teacher = augmented_exp_minus_energy(uniform_samples.unsqueeze(0), torch.t(teacher_Win).unsqueeze(2), torch.t(teacher_wout).unsqueeze(2), compute_mean=False)
            log_term_student = torch.log(part_mean_student)
            log_term_teacher = torch.log(part_mean_teacher)
            log_term_diff = float(log_term_teacher - log_term_student)
            unnorm_densities_student = unnormalized_densities(uniform_samples.unsqueeze(0), torch.t(Win).unsqueeze(2), torch.t(wout).unsqueeze(2))
            unnorm_densities_teacher = unnormalized_densities(uniform_samples.unsqueeze(0), torch.t(teacher_Win).unsqueeze(2), torch.t(teacher_wout).unsqueeze(2))
            energies_student = energies(uniform_samples.unsqueeze(0), torch.t(Win).unsqueeze(2), torch.t(wout).unsqueeze(2))
            energies_teacher = energies(uniform_samples.unsqueeze(0), torch.t(teacher_Win).unsqueeze(2), torch.t(teacher_wout).unsqueeze(2))
            avg_term_comp = (energies_student-energies_teacher)*unnorm_densities_teacher/part_mean_teacher
            avg_term_comp[torch.isnan(avg_term_comp)] = 0
            avg_term = float(torch.sum(avg_term_comp))
            if not np.isnan(-log_term_diff + avg_term):
                kl_reps.append(-log_term_diff + avg_term)
    kl_avg = np.mean(kl_reps)
    return kl_avg

def decreasing_stepsize(a,b,n_iteration):
    return a/(1+b*n_iteration)


def set_args_for_task_id(args, task_id):
    grid = {
        'model': ['f2', 'f1'],
        'd': [15],
        'n': [10, 30, 100, 300, 1000, 3000, 10000, 30000],
        'lr': [0.005, 0.05, 0.5],
        'wd': [0, 1e-8, 1e-5],
    }
    from itertools import product
    gridlist = list(dict(zip(grid.keys(), vals)) for vals in product(*grid.values()))

    print(f'task {task_id} out of {len(gridlist)}')
    assert task_id >= 1 and task_id <= len(gridlist), 'wrong task_id!'
    elem = gridlist[task_id - 1]
    for k, v in elem.items():
        setattr(args, k, v)


def set_args_for_task_id_rep(args, task_id):
    gridlist = []
    fields = ['model', 'd', 'target_neurons', 'seed', 'n', 'mcmc_samples', 'neurons', 'lr', 'wd', 'positive_weights', 'n_negative_weights']
    field_types = [str, int, int, int, int, int, int, float, float, to_bool, int]

    assert len(fields) == len(field_types)

    for line in open(os.path.join('res', args.name, 'hyper.txt')):
        chunks = line.strip()[:-4].split('_')
        dd = {}

        dd.update(dict(zip(fields,
            [ty(val) for ty, val in zip(field_types, chunks[-len(fields):])])))
        print(dd)

        for rep in range(10):
            ddr = dd.copy()
            ddr['exptrep'] = rep
            gridlist.append(ddr)

    print(f'task {task_id} out of {len(gridlist)}')
    assert task_id >= 1 and task_id <= len(gridlist), 'wrong task_id!'
    elem = gridlist[task_id - 1]
    for k, v in elem.items():
        setattr(args, k, v)
    args.name += '_rep{}'.format(args.exptrep)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MLE training')
    parser.add_argument('--name', default='mle', help='exp name')
    parser.add_argument('--model', default='f2', help='f1 or f2')
    parser.add_argument('--task_id', type=int, default=None, help='task_id for sweep jobs')
    parser.add_argument('--task_id_rep', type=int, default=None, help='task_id for sweep jobs for resampling')
    parser.add_argument('--exptrep', type=int, default=None, help='for experiment repetitions')
    parser.add_argument('--d', type=int, default=10, help='dimension of the data')
    parser.add_argument('--target_neurons', type=int, default=1, help='number of neurons in the teacher')
    parser.add_argument('--ball_radius', type=float, default=2., help='data radius')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--n', type=int, default=1000, help='number of train/test examples')
    parser.add_argument('--mcmc_samples', type=int, default=1000,
                        help='number of mcmc samples for computing gradients')
    parser.add_argument('--neurons', type=int, default=1000, help='number of neurons/width')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--lrb', type=float, default=0.05, help='learning rate b param')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--niter', type=int, default=151, help='number of iterations')
    parser.add_argument('--gd_updates_per_iter', type=int, default=10, help='number of gd updates per iteration')
    parser.add_argument('--eval_delta', type=int, default=10, help='how often to compute val/test metrics')
    parser.add_argument('--eval_unif_samples', type=int, default=100000,
                        help='number of uniform samples for evaluation')
    parser.add_argument('--eval_unif_samples_kl', type=int, default=200000, help='number of uniform samples for KL evaluation')
    parser.add_argument('--eval_reps', type=int, default=1,
                        help='number of averaging samples for cross entropy evaluation')
    parser.add_argument('--recompute_data', action='store_true', help='recompute teacher samples')
    parser.add_argument('--interactive', action='store_true', help='interactive, i.e. do not save results')
    parser.add_argument('--positive_weights', type=bool, default=True, help='If True, target_wout is positive')
    parser.add_argument('--allow_negative_weights', action='store_false', dest='positive_weights', help='makes positive_weights negative')
    parser.add_argument('--alpha', type=float, default=1e-3, help='initialization scaling')
    parser.add_argument('--n_negative_weights', type=int, default=-1, help='number of negative target neurons, -1 if random')

    args = parser.parse_args()

    if args.task_id is not None:
        set_args_for_task_id(args, args.task_id)
    if args.task_id_rep is not None:
        set_args_for_task_id_rep(args, args.task_id_rep)

    resdir = os.path.join('res', args.name)
    fname = os.path.join(resdir,
            f'{args.name}_{args.model}_{args.d}_{args.target_neurons}_{args.seed}_{args.n}_{args.mcmc_samples}_{args.neurons}_{args.lr}_{args.wd}_{args.positive_weights}_{args.n_negative_weights}.pkl')
    if not os.path.exists(resdir):
        os.makedirs(resdir)

    print('output:', fname)

    if os.path.exists(fname) and not args.interactive:
        print('results file already exists, skipping')
        sys.exit(0)

    device = torch.device("cpu")

    X, Xval, Xte = teacher_samples(args)

    assert args.n <= X.shape[0], 'sample size is larger than precomputed data'
    X = X[:args.n]
    X = torch.tensor(X).to(device)
    Xval = torch.tensor(Xval).to(device)
    Xte = torch.tensor(Xte).to(device)

    # Evaluate teacher cross entropy
    teacher_Win, teacher_wout = get_teacher(args.d, args.target_neurons, args.ball_radius, args.seed, args.positive_weights, args.n_negative_weights)
    print(teacher_wout)
    teacher_f, teacher_f_stuff = free_energy_sampling(teacher_Win, teacher_wout, n_uniform_samples=args.eval_unif_samples, reps=args.eval_reps, get_stuff=True)

    teacher_train_cea = cross_entropy_avgterm(teacher_Win, teacher_wout, X)
    teacher_val_cea = cross_entropy_avgterm(teacher_Win, teacher_wout, Xval)
    teacher_test_cea = cross_entropy_avgterm(teacher_Win, teacher_wout, Xte)

    teacher_train_ce = teacher_train_cea + teacher_f
    teacher_val_ce = teacher_val_cea + teacher_f
    teacher_test_ce = teacher_test_cea + teacher_f

    print(f'Teacher cross entropy (XE): train {teacher_train_ce}, val {teacher_val_ce}, test {teacher_test_ce}')

    Win = torch.randn(args.d, args.neurons)
    
    if args.model == 'f2':
        wout = torch.zeros(1, args.neurons)
        max_cosine = teacher_Win.t().matmul(Win / torch.norm(Win, dim=0).unsqueeze(0)).abs().max()
    elif args.model == 'f1':
        Win *= np.sqrt(args.alpha)
        wout = torch.randn(1, args.neurons) * np.sqrt(args.alpha)

    Win.to(device)
    wout.to(device)

    iters = []
    train_cross_entropy = []
    val_cross_entropy = []
    test_cross_entropy = []
    train_ce_incr = []
    val_ce_incr = []
    test_ce_incr = []
    free_energy_stuff = []
    f_sampling = []
    f_incr = []
    max_cosines = []
    Win_list = []
    wout_list = []
    if args.task_id_rep is not None:
        train_kl_div = []
        val_kl_div = []
        test_kl_div = []
        kl_div_2 = []

    F_est = 0.

    for t in range(args.niter):
        tstart = time.time()
        potential = lambda x: energy(x, Win, wout)

        Xmcmc_all = torch.tensor(get_samples_uniform_proposal(potential, args.d, n_samples=args.mcmc_samples, step=0.1, burn=2, burn_in=5000))
        Xmcmc_all.to(device)
        Xmcmc = Xmcmc_all[:args.mcmc_samples]
        dt_mcmc = time.time() - tstart

        lr = decreasing_stepsize(args.lr, args.lrb, t * args.gd_updates_per_iter / 10)

        energies_before = potential(Xmcmc_all)
        for k in range(args.gd_updates_per_iter):
            gradout, gradout_model = loss_gradient_weights(Win, wout, X, Xmcmc)
            if args.model == 'f1':
                gradin, gradin_model = loss_gradient_positions(Win, wout, X, Xmcmc)

            out_step = lr * args.neurons * (gradout.data + args.wd * wout.data)
            wout.sub_(out_step)
            if args.model == 'f1':  # both layers for F1
                in_step = 100. * lr * args.neurons * (gradin + args.wd * Win.data)
                Win.sub_(in_step)

        energies_after = potential(Xmcmc_all)
        # update free energy estimates
        F_est += torch.log(torch.mean(torch.exp(-(energies_after - energies_before))))

        print("Iteration", t, "done. ({:.2f} / {:.2f})".format(dt_mcmc, time.time() - tstart))
        if args.model == 'f1':
            print("F1 norm:", torch.norm(torch.norm(Win, p=2, dim=0).unsqueeze(0)*wout, p=1)/args.neurons)

        if t % args.eval_delta == 0:
            if args.task_id_rep is not None:
                kl_2 = compute_kl_a_posteriori(teacher_Win, teacher_wout, Win, wout, n_uniform_samples=30000, reps=5)
                ce_f, ce_f_stuff = free_energy_sampling(Win, wout, n_uniform_samples=args.eval_unif_samples_kl, reps=args.eval_reps, get_stuff=True)
            else:
                ce_f, ce_f_stuff = free_energy_sampling(Win, wout, n_uniform_samples=args.eval_unif_samples, reps=args.eval_reps, get_stuff=True)

            train_cea = cross_entropy_avgterm(Win, wout, X)
            val_cea = cross_entropy_avgterm(Win, wout, Xval)
            test_cea = cross_entropy_avgterm(Win, wout, Xte)

            train_ce = train_cea + ce_f
            val_ce = val_cea + ce_f
            test_ce = test_cea + ce_f
            train_ce_i = train_cea + F_est
            val_ce_i = val_cea + F_est
            test_ce_i = test_cea + F_est

            if args.task_id_rep is not None:
                diffe_f, diffe_f_stuff = free_energy_sampling(teacher_Win, teacher_wout, n_uniform_samples=args.eval_unif_samples_kl, reps=args.eval_reps, get_stuff=True)
                train_diffea = cross_entropy_avgterm(teacher_Win, teacher_wout, X)
                val_diffea = cross_entropy_avgterm(teacher_Win, teacher_wout, Xval)
                test_diffea = cross_entropy_avgterm(teacher_Win, teacher_wout, Xte)

                train_kl = train_ce - train_diffea - diffe_f
                val_kl = val_ce - val_diffea - diffe_f
                test_kl = test_ce - test_diffea - diffe_f

            if args.model == 'f1':
                max_cosine = teacher_Win.t().matmul(Win / torch.norm(Win, dim=0).unsqueeze(0)).abs().max()

            dt_tot = time.time() - tstart
            print(f'Cross entropy (XE): train {train_ce:.6f} (teacher train: {teacher_train_ce:.6f}), val {val_ce:.6f} (teacher val: {teacher_val_ce:.6f}), test {test_ce:.6f} (teacher test: {teacher_test_ce:.6f}). Cosine max {max_cosine}. ({dt_tot:.2f})', flush=True)

            iters.append(t)
            train_cross_entropy.append(train_ce)
            val_cross_entropy.append(val_ce)
            test_cross_entropy.append(test_ce)

            train_ce_incr.append(train_ce_i)
            val_ce_incr.append(val_ce_i)
            test_ce_incr.append(test_ce_i)

            max_cosines.append(max_cosine)
            free_energy_stuff.append(ce_f_stuff)

            f_sampling.append(ce_f)
            f_incr.append(F_est)

            Win_list.append(Win)
            wout_list.append(wout)

            if args.task_id_rep is not None:
                train_kl_div.append(train_kl)
                val_kl_div.append(val_kl)
                test_kl_div.append(test_kl)
                kl_div_2.append(kl_2)

            if not args.interactive and args.task_id_rep is None:
                res = {
                    'teacher_train_ce': teacher_train_ce,
                    'teacher_val_ce': teacher_val_ce,
                    'teacher_test_ce': teacher_test_ce,
                    'train_cross_entropy': train_cross_entropy,
                    'val_cross_entropy': val_cross_entropy,
                    'test_cross_entropy': test_cross_entropy,
                    'train_ce_incr': train_ce_incr,
                    'val_ce_incr': val_ce_incr,
                    'test_ce_incr': test_ce_incr,
                    'teacher_f_stuff': teacher_f_stuff,
                    'free_energy_stuff': free_energy_stuff,
                    'f_sampling': f_sampling,
                    'f_incr': f_incr,
                    'max_cosines': max_cosines,
                    'iters': iters,
                    'teacher_Win': teacher_Win,
                    'teacher_wout': teacher_wout,
                    'Win': Win_list,
                    'wout': wout_list
                }

                pickle.dump(res, open(fname, 'wb'))

            if not args.interactive and args.task_id_rep is not None:
                res = {
                    'teacher_train_ce': teacher_train_ce,
                    'teacher_val_ce': teacher_val_ce,
                    'teacher_test_ce': teacher_test_ce,
                    'train_cross_entropy': train_cross_entropy,
                    'val_cross_entropy': val_cross_entropy,
                    'test_cross_entropy': test_cross_entropy,
                    'train_kl_div': train_kl_div,
                    'val_kl_div': val_kl_div,
                    'test_kl_div': test_kl_div,
                    'kl_div_2': kl_div_2,
                    'train_ce_incr': train_ce_incr,
                    'val_ce_incr': val_ce_incr,
                    'test_ce_incr': test_ce_incr,
                    'teacher_f_stuff': teacher_f_stuff,
                    'free_energy_stuff': free_energy_stuff,
                    'f_sampling': f_sampling,
                    'f_incr': f_incr,
                    'max_cosines': max_cosines,
                    'iters': iters,
                    'teacher_Win': teacher_Win,
                    'teacher_wout': teacher_wout,
                    'Win': Win_list,
                    'wout': wout_list
                }

                pickle.dump(res, open(fname, 'wb'))
