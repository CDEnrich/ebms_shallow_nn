import numpy as np
import torch
import torch.nn as nn
import time

def log_Q(potential, z_prime, z, step):
    z.requires_grad_()
    grad = torch.autograd.grad(potential(z)[0], z)[0]
    return -(torch.norm(z_prime - z + step * grad, p=2, dim=1) ** 2) / (4 * step)


def get_samples(potential, dimension, n_samples=1000,
                step=0.1, burn=2, burn_in=3000, show_progress=False):
    #burn_in = 10000
    Z0 = torch.randn(1, dimension)
    Z0 = nn.functional.normalize(Z0, p=2, dim=1)
    Zi = Z0
    samples = []
    # samples = np.zeros((n_samples, dimension), dtype=np.float32)
    pbar = range(burn*n_samples + burn_in) #2 instead of 10
    reject = 0
    current_time = time.perf_counter()
    for i in pbar:
        if show_progress and i % 2000 == 0:
            print('.', end='', flush=True)
        Zi.requires_grad_()
        u = potential(Zi)[0]
        grad = torch.autograd.grad(u, Zi)[0]
        prop_Zi = Zi.detach() - step * grad + np.sqrt(2 * step) * torch.randn(1, dimension)
        prop_Zi = nn.functional.normalize(prop_Zi, p=2, dim=1)
        log_ratio = -potential(prop_Zi)[0] + potential(Zi)[0] + \
                    log_Q(potential, Zi, prop_Zi, step) - log_Q(potential, prop_Zi, Zi, step)
        #if i == 1 or i == 2:
        #    current_time_2 = time.perf_counter()
        #    print(current_time_2 - current_time)
        #    current_time = current_time_2
        if torch.rand(1) < torch.exp(log_ratio):
            Zi = prop_Zi.detach()
        else:
            reject += 1
        if i%burn == 0:
            samples.append(Zi.detach().numpy())
        # if i >= burn_in and i % burn == 0:
        #     idx = (i - burn_in) // burn
        #     samples[idx] = Zi.detach().numpy()
    print('mala rejections: {} out of {}'.format(reject, burn * n_samples + burn_in))
    return np.concatenate(samples, 0)[int(burn_in/burn):]
    # return samples
    
def get_samples_uniform_proposal(potential, dimension, n_samples=1000,
                step=0.1, burn=2, burn_in=3000, show_progress=False):
    #burn_in = 10000
    Z0 = torch.randn(1, dimension)
    Z0 = nn.functional.normalize(Z0, p=2, dim=1)
    Zi = Z0
    samples = []
    # samples = np.zeros((n_samples, dimension), dtype=np.float32)
    pbar = range(burn*n_samples + burn_in) #2 instead of 10
    reject = 0
    current_time = time.perf_counter()
    for i in pbar:
        if show_progress and i % 2000 == 0:
            print('.', end='', flush=True)
        prop_Zi = torch.randn(1, dimension)
        prop_Zi = nn.functional.normalize(prop_Zi, p=2, dim=1)
        log_ratio = -potential(prop_Zi)[0] + potential(Zi)[0]
        #if i == 1 or i == 2:
        #    current_time_2 = time.perf_counter()
        #    print(current_time_2 - current_time)
        #    current_time = current_time_2
        if torch.rand(1) < torch.exp(log_ratio):
            Zi = prop_Zi
        else:
            reject += 1
        if i%burn == 0:
            samples.append(Zi.detach().numpy())
        # if i >= burn_in and i % burn == 0:
        #     idx = (i - burn_in) // burn
        #     samples[idx] = Zi.detach().numpy()
    print('mala rejections: {} out of {}'.format(reject, burn * n_samples + burn_in))
    return np.concatenate(samples, 0)[int(burn_in/burn):]
    # return samples
