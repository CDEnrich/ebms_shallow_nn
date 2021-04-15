import argparse
import numpy as np
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import time

from mala import get_samples
from mle import cross_entropy_avgterm, free_energy_sampling, compute_kl_a_posteriori
from util import get_teacher, teacher_samples, to_bool

TEACHER_BURNIN = 10000
TEACHER_BURN = 4

def score_function(Win,wout,X):
    neurons = Win.shape[0]
    score = -torch.sum((torch.sign(torch.matmul(torch.transpose(Win,1,2), \
                                                           torch.transpose(X,1,2)))*0.5 + 0.5) \
                                  *Win*wout, dim=0)/neurons
    return score-torch.sum((X.squeeze(0).t()*score), dim=0)*X.squeeze(0).t()

def f2_kernel_evaluation(X):
    dimension = X.shape[1]
    inner_prod = torch.matmul(X,X.t()).fill_diagonal_(fill_value = 1)
    return ((np.pi-torch.acos(inner_prod))*inner_prod \
            + torch.sqrt(1-inner_prod*inner_prod))/(2*np.pi*(dimension+1))

def f2_kernel_derivatives(X):
    dimension = samples.shape[1]
    inner_prod = torch.matmul(X,X.t()).fill_diagonal_(fill_value = 1)
    return ((np.pi-torch.acos(inner_prod))*inner_prod \
            + 2*(inner_prod/torch.sqrt(1-inner_prod*inner_prod)).fill_diagonal_(fill_value  = 0)).unsqueeze(2)*X.unsqueeze(0)/(2*np.pi*(dimension+1))

def rbf_kernel_evaluation(X):
    dimension = X.shape[1]
    norm_sq_differences = 1 - torch.matmul(X,X.t())
    return (torch.exp(-norm_sq_differences))

def rbf_kernel_derivatives(X):
    dimension = X.shape[1]
    norm_sq_differences = 1 - torch.matmul(X,X.t())
    gradient = (torch.exp(-norm_sq_differences)).unsqueeze(2)*(X.unsqueeze(0)-X.unsqueeze(1))
    return gradient - (torch.sum(gradient*X.unsqueeze(1), dim=2)).unsqueeze(2)*X.unsqueeze(1)

def rbf_kernel_tr_term(X):
    dimension = X.shape[1]
    norm_sq_differences = 1 - torch.matmul(X,X.t())
    y_projected_x = X.unsqueeze(0) - (torch.sum(X.unsqueeze(0)*X.unsqueeze(1), dim=2)).unsqueeze(2)*X.unsqueeze(1)
    print("y_projected_x", y_projected_x.shape)
    x_projected_y = X.unsqueeze(1) - (torch.sum(X.unsqueeze(1)*X.unsqueeze(0), dim=2)).unsqueeze(2)*X.unsqueeze(0)
    print("x_projected_y", x_projected_y.shape)
    first_term = (torch.exp(-norm_sq_differences))*torch.sum(y_projected_x*x_projected_y, dim=2)
    print("first_term", first_term.shape)
    second_term = (torch.exp(-norm_sq_differences))*(dimension-2+(torch.matmul(X,X.t()))**2)
    print("second_term", second_term.shape)
    return first_term + second_term

def KSD_computation(Win,wout,X,kernel_eval,kernel_der,kernel_tr):
    n_samples = X.shape[0]
    data_dim = X.shape[1]
    score = score_function(Win.unsqueeze(2),wout.unsqueeze(2),X.unsqueeze(0))
    score_corrected = score - (data_dim-1)*X.t()
    first_term = torch.sum((torch.matmul(torch.t(score_corrected), score_corrected)*kernel_eval).fill_diagonal_(fill_value = 0))/(n_samples*(n_samples-1))
    second_term = 2*torch.sum(torch.sum(kernel_der*score_corrected.t().unsqueeze(0), dim=2).fill_diagonal_(fill_value = 0))/(n_samples*(n_samples-1))
    third_term = torch.sum(kernel_tr.fill_diagonal_(fill_value = 0))/(n_samples*(n_samples-1))
    fourth_term = torch.sum((torch.matmul(torch.t(score), score)*kernel_eval).fill_diagonal_(fill_value = 0))/(n_samples*(n_samples-1))
    stein_identity = torch.norm(torch.mean(score_corrected.t().unsqueeze(0)*kernel_eval.unsqueeze(2) + torch.transpose(kernel_der,0,1), dim=1), p=1)/(n_samples*data_dim)
    stein_identity_first = torch.norm(torch.mean(score_corrected.t().unsqueeze(0)*kernel_eval.unsqueeze(2), dim=1), p=1)/(n_samples*data_dim)
    stein_identity_second = torch.norm(torch.mean(torch.transpose(kernel_der,0,1), dim=1), p=1)/(n_samples*data_dim)
    stein_identity_third = torch.norm(torch.mean(score.t().unsqueeze(0)*kernel_eval.unsqueeze(2), dim=1), p=1)/(n_samples*data_dim)
    return (first_term + second_term + third_term, first_term, second_term, third_term, fourth_term, stein_identity, stein_identity_first, stein_identity_second, stein_identity_third)

def decreasing_stepsize(a,b,n_iteration):
    return a/(1+b*n_iteration)

def set_args_for_task_id(args, task_id):
    grid = {
        'model': ['f2', 'f1'],
        'd': [15],
        'n': [10, 30, 100, 300, 1000, 3000, 10000, 30000],
        'lr': [0.05, 0.2, 0.5],
        'wd': [0, 1e-9, 1e-7],
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
    fields = ['model', 'd', 'target_neurons', 'seed', 'n', 'neurons', 'lr', 'wd', 'positive_weights', 'n_negative_weights']
    field_types = [str, int, int, int, int, int, float, float, to_bool, int]
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
    parser = argparse.ArgumentParser(description='KSD training')
    parser.add_argument('--name', default='ksd', help='exp name')
    parser.add_argument('--model', default='f2', help='f1 or f2')
    parser.add_argument('--task_id', type=int, default=None, help='task_id for sweep jobs')
    parser.add_argument('--task_id_rep', type=int, default=None, help='task_id for sweep jobs for resampling')
    parser.add_argument('--d', type=int, default=10, help='dimension of the data')
    parser.add_argument('--target_neurons', type=int, default=1, help='number of neurons in the teacher')
    parser.add_argument('--ball_radius', type=float, default=2., help='data radius')
    parser.add_argument('--seed', type=int, default=32, help='random seed')
    parser.add_argument('--exptrep', type=int, default=None, help='for experiment repetitions')
    parser.add_argument('--n', type=int, default=1000, help='number of train/test examples')
    parser.add_argument('--neurons', type=int, default=1000, help='number of neurons/width')
    parser.add_argument('--lr', type=float, default=1.0, help='learning rate') #0.01
    parser.add_argument('--lrb', type=float, default=0.01, help='learning rate b param')
    parser.add_argument('--wd', type=float, default=1e-6, help='weight decay') #1e-4
    parser.add_argument('--niter', type=int, default=151, help='number of iterations')
    parser.add_argument('--eval_delta', type=int, default=10, help='how often to compute val/test metrics')
    parser.add_argument('--kernel', default='rbf', help='kernel used: rbf/f2')
    parser.add_argument('--recompute_data', action='store_true', help='recompute teacher samples')
    parser.add_argument('--interactive', action='store_true', help='interactive, i.e. do not save results')
    parser.add_argument('--eval_unif_samples', type=int, default=100000,
                        help='number of uniform samples for evaluation')
    parser.add_argument('--eval_unif_samples_kl', type=int, default=200000, help='number of uniform samples for KL evaluation')
    parser.add_argument('--positive_weights', type=bool, default=True, help='If True, target_wout is positive')
    parser.add_argument('--allow_negative_weights', action='store_false', dest='positive_weights', help='makes positive_weights negative')
    parser.add_argument('--eval_reps', type=int, default=1,
                        help='number of averaging samples for cross entropy evaluation')
    parser.add_argument('--alpha', type=float, default=1e-3, help='initialization scaling')
    parser.add_argument('--n_negative_weights', type=int, default=-1, help='number of negative target neurons, -1 if random')
    
    args = parser.parse_args()
    
    if args.task_id is not None:
        set_args_for_task_id(args, args.task_id)
    if args.task_id_rep is not None:
        set_args_for_task_id_rep(args, args.task_id_rep)

    resdir = os.path.join('res', args.name)
    fname = os.path.join(resdir, f'{args.name}_{args.model}_{args.d}_{args.target_neurons}_{args.seed}_{args.n}_{args.neurons}_{args.lr}_{args.wd}_{args.positive_weights}_{args.n_negative_weights}.pkl')

    print('output:', fname)
    if os.path.exists(fname) and not args.interactive:
        print('results file already exists, skipping')
        sys.exit(0)
    
    device = torch.device("cpu")

    if not args.interactive and not os.path.exists(resdir):
        os.makedirs(resdir)

    X, Xval, Xte = teacher_samples(args)

    assert args.n <= X.shape[0], 'sample size is larger than precomputed data'
    assert (args.kernel == "rbf" or args.kernel == "f2"), 'kernel not available' 
    X = X[:args.n]
    X = torch.tensor(X).to(device)
    Xval = torch.tensor(Xval).to(device)
    Xte = torch.tensor(Xte).to(device)

    # reduce val/test sizes for now...
    Xval = Xval[:5000]
    Xte = Xte[:5000]
    
    def plot():
        plt.clf()
        cosine = teacher_Win[:,0].dot(teacher_Win[:,1])

        theta = np.arccos(cosine)
        first = teacher_Win[:,0]
        second = nn.functional.normalize(teacher_Win[:,1] - cosine * first, p=2, dim=0)

        if args.n_negative_weights == 2:
            plt.plot([0., 1.], [0., 0.], color='b')
            plt.plot([0., cosine], [0., teacher_Win[:,1].dot(second)], color='b')
        elif args.n_negative_weights == 1:
            plt.plot([0., 1.], [0., 0.], color='b')
            plt.plot([0., cosine], [0., teacher_Win[:,1].dot(second)], color='c')
        elif args.n_negative_weights == 0:
            plt.plot([0., 1.], [0., 0.], color='c')
            plt.plot([0., cosine], [0., teacher_Win[:,1].dot(second)], color='c')
        

        normWin = nn.functional.normalize(Win, p=2, dim=0)
        norms = torch.norm(Win, p=2, dim=0)
        norms *= wout[0,:]
        for i in range(Win.shape[1]):
            plt.plot([0., normWin[:,i].dot(first)], [0., normWin[:,i].dot(second)], color='g' if norms[i] > 0 else 'r', alpha=float(abs(norms[i]) / norms.abs().max()))


        plt.show()
        plt.pause(1.)
    
    print('evaluating kernels')
    if args.kernel=='rbf':
        kernel_eval_X = rbf_kernel_evaluation(X)
        kernel_der_X = rbf_kernel_derivatives(X)
        kernel_tr_X = rbf_kernel_tr_term(X)
        kernel_eval_Xval = rbf_kernel_evaluation(Xval)
        kernel_der_Xval = rbf_kernel_derivatives(Xval)
        kernel_tr_Xval = rbf_kernel_tr_term(Xval)
        kernel_eval_Xte = rbf_kernel_evaluation(Xte)
        kernel_der_Xte = rbf_kernel_derivatives(Xte)
        kernel_tr_Xte = rbf_kernel_tr_term(Xte)
        
    else:
        kernel_eval_X = f2_kernel_evaluation(X)
        kernel_der_X = f2_kernel_derivatives(X)
        kernel_eval_Xval = f2_kernel_evaluation(Xval)
        kernel_der_Xval = f2_kernel_derivatives(Xval)
        kernel_eval_Xte = f2_kernel_evaluation(Xte)
        kernel_der_Xte = f2_kernel_derivatives(Xte)
    
    # Evaluate teacher KSD
    print('evaluating teacher')
    teacher_Win, teacher_wout = get_teacher(args.d, args.target_neurons, args.ball_radius, args.seed, args.positive_weights, args.n_negative_weights)
    teacher_train_KSD = KSD_computation(teacher_Win.t(), teacher_wout.t(), X, kernel_eval_X, kernel_der_X, kernel_tr_X)
    teacher_val_KSD = KSD_computation(teacher_Win.t(), teacher_wout.t(), Xval, kernel_eval_Xval, kernel_der_Xval, kernel_tr_Xval)
    teacher_test_KSD = KSD_computation(teacher_Win.t(), teacher_wout.t(), Xte, kernel_eval_Xte, kernel_der_Xte, kernel_tr_Xte)

    teacher_f, teacher_f_stuff = free_energy_sampling(teacher_Win, teacher_wout, n_uniform_samples=args.eval_unif_samples, reps=args.eval_reps, get_stuff=True)

    teacher_train_cea = cross_entropy_avgterm(teacher_Win, teacher_wout, X)
    teacher_val_cea = cross_entropy_avgterm(teacher_Win, teacher_wout, Xval)
    teacher_test_cea = cross_entropy_avgterm(teacher_Win, teacher_wout, Xte)

    teacher_train_ce = teacher_train_cea + teacher_f
    teacher_val_ce = teacher_val_cea + teacher_f
    teacher_test_ce = teacher_test_cea + teacher_f

    if args.target_neurons == 1:
        cosine_train = (torch.dot(teacher_Win.squeeze(1),torch.mean(X, dim=0))/(torch.norm(teacher_Win.squeeze(1))*torch.norm(torch.mean(X, dim=0)))).item()
        cosine_val = (torch.dot(teacher_Win.squeeze(1),torch.mean(Xval, dim=0))/(torch.norm(teacher_Win.squeeze(1))*torch.norm(torch.mean(Xval, dim=0)))).item()
        cosine_test = (torch.dot(teacher_Win.squeeze(1),torch.mean(Xte, dim=0))/(torch.norm(teacher_Win.squeeze(1))*torch.norm(torch.mean(Xte, dim=0)))).item()
    
    print(f'Teacher KSD: train {teacher_train_KSD[0]}, val {teacher_val_KSD[0]}, test {teacher_test_KSD[0]}')
    print(f'Teacher KSD first term: train {teacher_train_KSD[1]}, val {teacher_val_KSD[1]}, test {teacher_test_KSD[1]}')
    print(f'Teacher KSD second term: train {teacher_train_KSD[2]}, val {teacher_val_KSD[2]}, test {teacher_test_KSD[2]}')
    print(f'Teacher KSD third term: train {teacher_train_KSD[3]}, val {teacher_val_KSD[3]}, test {teacher_test_KSD[3]}')
    print(f'Teacher KSD fourth term: train {teacher_train_KSD[4]}, val {teacher_val_KSD[4]}, test {teacher_test_KSD[4]}')
    print(f'Teacher KSD Stein identity: train {teacher_train_KSD[5]}, val {teacher_val_KSD[5]}, test {teacher_test_KSD[5]}')
    print(f'Teacher KSD Stein identity first: train {teacher_train_KSD[6]}, val {teacher_val_KSD[6]}, test {teacher_test_KSD[6]}')
    print(f'Teacher KSD Stein identity second: train {teacher_train_KSD[7]}, val {teacher_val_KSD[7]}, test {teacher_test_KSD[7]}')
    print(f'Teacher KSD Stein identity third: train {teacher_train_KSD[8]}, val {teacher_val_KSD[8]}, test {teacher_test_KSD[8]}')
    print(f'Teacher cross entropy (XE): train {teacher_train_ce}, val {teacher_val_ce}, test {teacher_test_ce}')
    print(f'Teacher output weight: {teacher_wout}')
    if args.target_neurons == 1:
        print(f'Cosines between expectation and target: train {cosine_train}, val {cosine_val}, test {cosine_test}')
    
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
    train_KSD, val_KSD, test_KSD = [], [], []
    train_cross_entropy = []
    val_cross_entropy = []
    test_cross_entropy = []
    free_energy_stuff = []
    max_cosines = []
    Win_list = []
    wout_list = []
    if args.task_id_rep is not None:
        train_kl_div = []
        val_kl_div = []
        test_kl_div = []
        kl_div_2 = []
    
    for t in range(args.niter):
        tstart = time.time()

        lr = decreasing_stepsize(args.lr, args.lrb, t)
        
        if args.model == 'f1':
            Win.requires_grad_()
            wout.requires_grad_()
            KSD_f1 = lambda Win, wout: KSD_computation(Win.t(), wout.t(), X, kernel_eval_X, kernel_der_X, kernel_tr_X)
            grad = torch.autograd.grad(KSD_f1(Win, wout)[0], [Win, wout])
            gradin = grad[0]
            gradout = grad[1]
            Win.detach_()
            wout.detach_()
            wout.sub_(lr * args.neurons * (gradout.data + args.wd * wout.data))
            Win.sub_(25 * lr * args.neurons * (gradin.data + args.wd * Win.data))
                           
        if args.model == 'f2':
            wout.requires_grad_()
            KSD_f2 = lambda wout: KSD_computation(Win.t(), wout.t(), X, kernel_eval_X, kernel_der_X, kernel_tr_X)
            gradout = torch.autograd.grad(KSD_f2(wout)[0], wout)[0]
            wout.detach_()
            wout.sub_(lr * args.neurons * (gradout.data + args.wd * wout.data))

        print("Iteration", t, "done. {:.2f}".format(time.time() - tstart))
        
        if args.model == 'f1':
            
            print("F1 norm:", torch.norm(torch.norm(Win, p=2, dim=0).unsqueeze(0)*wout, p=1)/args.neurons)

        if t % args.eval_delta == 0:
            train_kernel_stein = KSD_computation(Win.t(), wout.t(), X, kernel_eval_X, kernel_der_X, kernel_tr_X)
            val_kernel_stein = KSD_computation(Win.t(), wout.t(), Xval, kernel_eval_Xval, kernel_der_Xval, kernel_tr_Xval)
            test_kernel_stein = KSD_computation(Win.t(), wout.t(), Xte, kernel_eval_Xte, kernel_der_Xte, kernel_tr_Xte)
            
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
            print(f'KSD: train {train_kernel_stein[0]:.6f} (teacher train: {teacher_train_KSD[0]:.6f}), val {val_kernel_stein[0]:.6f} (teacher val: {teacher_val_KSD[0]:.6f}), test {test_kernel_stein[0]:.6f} (teacher test: {teacher_test_KSD[0]:.6f}). Cosine max {max_cosine:.6f}. ({dt_tot:.2f})', flush=True)
            print(f'Cross entropy: train {train_ce:.6f} (teacher train: {teacher_train_ce:.6f}), val {val_ce:.6f} (teacher val: {teacher_val_ce:.6f}), test {test_ce:.6f} (teacher test: {teacher_test_ce:.6f}).', flush=True)

            iters.append(t)
            train_KSD.append(train_kernel_stein[0])
            val_KSD.append(val_kernel_stein[0])
            test_KSD.append(test_kernel_stein[0])
            train_cross_entropy.append(train_ce)
            val_cross_entropy.append(val_ce)
            test_cross_entropy.append(test_ce)
            free_energy_stuff.append(ce_f_stuff)
            max_cosines.append(max_cosine)

            Win_list.append(Win)
            wout_list.append(wout)

            if args.task_id_rep is not None:
                train_kl_div.append(train_kl)
                val_kl_div.append(val_kl)
                test_kl_div.append(test_kl)
                kl_div_2.append(kl_2)

            if not args.interactive and args.task_id_rep is None:
                res = {
                    'teacher_train_KSD': teacher_train_KSD,
                    'teacher_val_KSD': teacher_val_KSD,
                    'teacher_test_KSD': teacher_test_KSD,
                    'train_KSD': train_KSD,
                    'val_KSD': val_KSD,
                    'test_KSD': test_KSD,
                    'teacher_train_ce': teacher_train_ce,
                    'teacher_val_ce': teacher_val_ce,
                    'teacher_test_ce': teacher_test_ce,
                    'train_cross_entropy': train_cross_entropy,
                    'val_cross_entropy': val_cross_entropy,
                    'test_cross_entropy': test_cross_entropy,
                    'free_energy_stuff': free_energy_stuff,
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
                    'teacher_train_KSD': teacher_train_KSD,
                    'teacher_val_KSD': teacher_val_KSD,
                    'teacher_test_KSD': teacher_test_KSD,
                    'train_KSD': train_KSD,
                    'val_KSD': val_KSD,
                    'test_KSD': test_KSD,
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
                    'free_energy_stuff': free_energy_stuff,
                    'max_cosines': max_cosines,
                    'iters': iters,
                    'teacher_Win': teacher_Win,
                    'teacher_wout': teacher_wout,
                    'Win': Win_list,
                    'wout': wout_list
                }

                pickle.dump(res, open(fname, 'wb'))
