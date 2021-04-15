import numpy as np
import os
import pickle
import torch
import torch.nn as nn


def energy(x, Win, wout):
    neurons = Win.shape[1]
    return torch.sum(nn.functional.relu(torch.matmul(x, Win)) * wout, dim=1)/neurons

def energy_sum(x, Win, wout):
    neurons = Win.shape[1]
    return torch.sum(nn.functional.relu(torch.matmul(x, Win)) * wout)/neurons

def get_teacher(d, target_neurons, ball_radius, seed, positive_weights, n_negative_weights):
    fname = os.path.join('model', f'teacher_{d}_{target_neurons}_{ball_radius}_{seed}_{positive_weights}_{n_negative_weights}.pkl')

    if os.path.exists(fname):
        target_Win, target_wout = pickle.load(open(fname, 'rb'))
    else:
        torch.manual_seed(seed + 5)
        target_Win = torch.randn(d, target_neurons)
        target_Win = torch.nn.functional.normalize(target_Win, p=2, dim=0)
        if positive_weights:
            target_wout = ball_radius*torch.ones([1, target_neurons])/target_neurons
        elif n_negative_weights != -1:
            target_wout = ball_radius*torch.cat((-torch.ones([1, n_negative_weights]),torch.ones([1, target_neurons-n_negative_weights])),1)/target_neurons
        else:
            target_wout = ball_radius*(torch.randint(0,2,(1, target_neurons))*2.0-1.0) / target_neurons

        if not os.path.exists('model'):
            os.makedirs('model')
        pickle.dump((target_Win, target_wout), open(fname, 'wb'))

    return target_Win, target_wout

def teacher_samples(args):
    target_neurons = args.target_neurons
    d = args.d
    ball_radius = args.ball_radius

    fname = os.path.join('data', f'data_{d}_{target_neurons}_{ball_radius}_{args.seed}_{args.positive_weights}_{args.n_negative_weights}.pkl')

    if args.exptrep is None and not args.recompute_data and os.path.exists(fname):
        Xtr, Xval, Xte = pickle.load(open(fname, 'rb'))
    else:
        if args.exptrep is None:
            print('Computing teacher samples (30k/20k/20k split)...', end='')
        else:
            print('Computing teacher samples (30k/20k/100k split)...', end='')
        target_Win, target_wout = get_teacher(d, target_neurons, ball_radius, args.seed, args.positive_weights, args.n_negative_weights)
        target_potential = lambda x: energy(x, target_Win, target_wout)

        if args.exptrep is not None:  # different seeds if expt repetition
            torch.manual_seed(args.seed + 1 + args.exptrep)
        else:
            torch.manual_seed(args.seed)

        print(target_wout)
        if args.exptrep is None:
            n_samples = 1000000
        else:
            n_samples = 5000000
        X = torch.randn(n_samples,d)
        X = torch.nn.functional.normalize(X, p=2, dim=1)
        X_energy = energy(X, target_Win, target_wout)
        if args.positive_weights:
            X_density = torch.exp(-X_energy)
        else:
            n_gd_samples = 10000
            gd_particles = torch.randn(n_gd_samples,d)
            gd_particles = torch.nn.functional.normalize(gd_particles, p=2, dim=1)
            for gd_iteration in range(5000):
                gd_particles.requires_grad_()
                fun_value = energy_sum(gd_particles, target_Win, target_wout)
                gradient = torch.autograd.grad(fun_value, gd_particles)[0]
                gd_particles.detach_()
                gd_particles.sub_(0.2*gradient)
                gd_particles = torch.nn.functional.normalize(gd_particles, p=2, dim=1)
                if gd_iteration%500==0:
                    final_values = energy(gd_particles, target_Win, target_wout)
                    min_energy = torch.min(final_values)
                    print("Minimum energy precomputation, iteration", gd_iteration, min_energy)
            final_values = energy(gd_particles, target_Win, target_wout)
            min_energy = torch.min(final_values)
            print("Minimum energy", min_energy)
            X_density = torch.exp(-X_energy)*np.exp(min_energy)
        acceptance_vector = torch.bernoulli(X_density)
        print(torch.norm(X_density, p=1), torch.norm(acceptance_vector, p=1))
        accepted_rows = []
        for i in range(n_samples):
            if acceptance_vector[i] == 1:
                accepted_rows.append(i)
        accepted_rows_tensor = torch.tensor(accepted_rows).unsqueeze(1).expand([len(accepted_rows),d])
        X = torch.gather(X, 0, accepted_rows_tensor)
        print('done')

        if args.exptrep is None:
            X = X[torch.randperm(70000).numpy(),:]  # shuffle data

            Xtr = X[:30000]
            Xval = X[30000:50000]
            Xte = X[50000:]

        else:
            X = X[torch.randperm(150000).numpy(),:]  # shuffle data

            Xtr = X[:30000]
            Xval = X[30000:50000]
            Xte = X[50000:]

        if not os.path.exists('data'):
            os.makedirs('data')
        if args.exptrep is None: # do not save reps
            pickle.dump((X, Xval, Xte), open(fname, 'wb'))

    return Xtr, Xval, Xte


def to_bool(s):
    return (s == 'True')
        