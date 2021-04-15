import argparse
import numpy as np
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import time

from mala import get_samples
from mle import cross_entropy_avgterm, free_energy_sampling
from util import get_teacher, teacher_samples, to_bool

TEACHER_BURNIN = 10000
TEACHER_BURN = 4

def score_function(Win,wout,X):
    neurons = Win.shape[0]
    score = -torch.sum((torch.sign(torch.matmul(torch.transpose(Win,1,2), \
                                                           torch.transpose(X,1,2)))*0.5 + 0.5) \
                                  *Win*wout, dim=0)/neurons
    return score-torch.sum((X.squeeze(0).t()*score), dim=0)*X.squeeze(0).t()

def h_function(Whin,whout,X):
    d = X.shape[1]
    n = X.shape[0]
    h_neurons = int(Whin.shape[0]/d)
    output = torch.zeros(d,n)
    for i in range(d):
        output[i,:] = torch.sum(nn.functional.relu(torch.matmul(X,Whin[(h_neurons*i):(h_neurons*(i+1)),:].t()))* \
                                                   whout[(h_neurons*i):(h_neurons*(i+1)),:].t(), dim=1) #/h_neurons
    return output

def h_function_comp(Whin,whout,X):
    d = X.shape[1]
    n = X.shape[0]
    neurons = int(Whin.shape[0])
    output = torch.sum(nn.functional.relu(torch.matmul(X,Whin.t()))*whout.t(), dim=1) 
    return output

def expectation_divergence_h(Whin,whout,X):
    d = X.shape[2]
    n = X.shape[1]
    h_neurons = int(Whin.shape[0]/d)
    output = torch.zeros(n)
    for i in range(d):
        v = torch.sum((torch.sign(torch.matmul(torch.transpose(Whin[(h_neurons*i):(h_neurons*(i+1)),:,:],1,2), \
                                                           torch.transpose(X,1,2)))*0.5 + 0.5) \
                                  *Whin[(h_neurons*i):(h_neurons*(i+1)),:,:] \
                      *whout[(h_neurons*i):(h_neurons*(i+1)),:,:], dim=0) #/h_neurons
        output = output + v[i,:] -torch.sum((X.squeeze(0).t()*v), dim=0)*(X.squeeze(0).t()[i,:])
    return torch.mean(output)

def expectation_divergence_h_comp(Whin,whout,X,i):
    d = X.shape[2]
    n = X.shape[1]
    neurons = int(Whin.shape[0])
    output = torch.zeros(n)
    v = torch.sum((torch.sign(torch.matmul(torch.transpose(Whin,1,2),torch.transpose(X,1,2)))*0.5 + 0.5)*Whin*whout, dim=0)
    output = output + v[i,:] -torch.sum((X.squeeze(0).t()*v), dim=0)*(X.squeeze(0).t()[i,:])
    return torch.mean(output)

def decreasing_stepsize(a,b,n_iteration):
    return a/(1+b*n_iteration)

def expectation_tr_stein_operator(Win,wout,Whin,whout,X):
    d = X.shape[1]
    h_neurons = int(Whin.shape[0]/d)
    score = score_function(Win.unsqueeze(2),wout.unsqueeze(2),X.unsqueeze(0))
    score_corrected = score - (d-1)*X.t()
    h_fun = h_function(Whin,whout,X) 
    mean_div_h = expectation_divergence_h(Whin.unsqueeze(2),whout.unsqueeze(2),X.unsqueeze(0)) 
    return (torch.mean(torch.sum(score_corrected*h_fun, dim=0)) + mean_div_h, 
            torch.mean(torch.sum(score_corrected*h_fun, dim=0)), mean_div_h)

def expectation_tr_stein_operator_comp(Win,wout,Whin,whout,X,i):
    d = X.shape[1]
    score = score_function(Win.unsqueeze(2),wout.unsqueeze(2),X.unsqueeze(0))
    score_corrected = score - (d-1)*X.t()
    score_corrected_i = score_corrected[i,:]
    h_fun = h_function_comp(Whin,whout,X) 
    mean_div_h = expectation_divergence_h_comp(Whin.unsqueeze(2),whout.unsqueeze(2),X.unsqueeze(0),i) 
    return (torch.mean(score_corrected_i*h_fun) + mean_div_h, 
            torch.mean(score_corrected_i*h_fun), mean_div_h)

def f1_stein_discrepancy(Win,wout,X,args):
    print("F1 Stein Discrepancy computation")
    tstart = time.time()
    dim_values = np.zeros(args.d) 
    for dim in range(args.d):
        Whin = torch.randn(args.d, args.neurons)
        whout = torch.randn(1, args.neurons)
        Whin.div_(torch.sqrt(torch.norm(whout.data*torch.norm(Whin.data, p=2, dim=0).unsqueeze(0), p=1)))
        whout.div_(torch.sqrt(torch.norm(whout.data*torch.norm(Whin.data, p=2, dim=0).unsqueeze(0), p=1)))
        for t in range(args.niter_stein_evaluation):
            lr = decreasing_stepsize(args.lr_stein, args.lrb_stein, t)
            expected_stein = lambda Whin, whout: expectation_tr_stein_operator_comp(Win.t(),wout.t(),
                                                                                        Whin.t(),whout.t(),X,dim)
            Whin.requires_grad_()
            whout.requires_grad_()
            fun_value = expected_stein(Whin, whout)[0]
            grad = torch.autograd.grad(fun_value, [Whin, whout])
            gradhin = grad[0]
            gradhout = grad[1]
            Whin.detach_()
            whout.detach_()
            Whin.add_(lr * gradhin.data) 
            whout.add_(lr * gradhout.data) 
            Whin.div_(torch.sqrt(torch.norm(whout.data*torch.norm(Whin.data, p=2, dim=0).unsqueeze(0), p=1)))
            whout.div_(torch.sqrt(torch.norm(whout.data*torch.norm(Whin.data, p=2, dim=0).unsqueeze(0), p=1)))
            if t%20 == 0:
                print(t, fun_value)
        dim_values[dim] = expectation_tr_stein_operator_comp(Win.t(),wout.t(),Whin.t(),whout.t(),X,dim)[0]
        print("dimension", dim, "done", time.time()-tstart, dim_values[dim])
    f1_stein_discrepancy = np.linalg.norm(dim_values)
    return f1_stein_discrepancy

def set_args_for_task_id(args, task_id):
    grid = {
        'model': ['f2', 'f1'],
        'd': [10],
        'n': [10, 30, 100, 300, 1000, 3000, 10000],
        'lr': [0.1, 0.5, 2.5],
        'wd': [1e-10, 1e-8, 1e-6],
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
    parser = argparse.ArgumentParser(description='F1 SD training')
    parser.add_argument('--name', default='f1_SD', help='exp name')
    parser.add_argument('--model', default='f2', help='f1 or f2')
    parser.add_argument('--task_id', type=int, default=None, help='task_id for sweep jobs')
    parser.add_argument('--task_id_rep', type=int, default=None, help='task_id for sweep jobs for resampling')
    parser.add_argument('--exptrep', type=int, default=None, help='for experiment repetitions')
    parser.add_argument('--d', type=int, default=10, help='dimension of the data')
    parser.add_argument('--target_neurons', type=int, default=1, help='number of neurons in the teacher')
    parser.add_argument('--ball_radius', type=float, default=2., help='data radius')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--n', type=int, default=1000, help='number of train examples')
    parser.add_argument('--n_test_val', type=int, default=2000, help='number of val/test examples')
    parser.add_argument('--neurons', type=int, default=1000, help='number of neurons/width')
    parser.add_argument('--lr', type=float, default=5.0, help='learning rate') #0.2 #5.0
    parser.add_argument('--lrb', type=float, default=0.01, help='learning rate b param')
    parser.add_argument('--lr_stein', type=float, default=1.0, help='learning rate') #5.0
    parser.add_argument('--lrb_stein', type=float, default=0.05, help='learning rate b param')
    parser.add_argument('--wd', type=float, default=1e-7, help='weight decay for input weights (probably unimportant)') #1e-4
    parser.add_argument('--niter', type=int, default=151, help='number of iterations')
    parser.add_argument('--niter_stein_evaluation', type=int, default=101, help='number of iterations for Stein evaluation')
    parser.add_argument('--niter_stein_training', type=int, default=25, help='number of iterations for Stein training')
    parser.add_argument('--eval_delta', type=int, default=10, help='how often to compute val/test metrics')
    parser.add_argument('--recompute_data', action='store_true', help='recompute teacher samples')
    parser.add_argument('--interactive', action='store_true', help='interactive, i.e. do not save results')
    parser.add_argument('--eval_unif_samples', type=int, default=100000,
                        help='number of uniform samples for evaluation')
    parser.add_argument('--eval_unif_samples_kl', type=int, default=200000, help='number of uniform samples for KL evaluation')
    parser.add_argument('--eval_reps', type=int, default=1,
                        help='number of averaging samples for cross entropy evaluation')
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
    fname = os.path.join(resdir, f'{args.name}_{args.model}_{args.d}_{args.target_neurons}_{args.seed}_{args.n}_{args.neurons}_{args.lr}_{args.wd}_{args.positive_weights}_{args.n_negative_weights}.pkl')
    print('output:', fname, flush=True)

    if not args.interactive and not os.path.exists(resdir):
        os.makedirs(resdir)

    if os.path.exists(fname) and not args.interactive:
        print('results file already exists, skipping')
        sys.exit(0)
    
    device = torch.device("cpu")

    X, Xval, Xte = teacher_samples(args)
    
    assert args.n <= X.shape[0], 'sample size is larger than precomputed data'
    X = X[:args.n]
    Xval = Xval[:args.n_test_val]
    Xte = Xte[:args.n_test_val]
    X = X.to(device)
    Xval = Xval.to(device)
    Xte = Xte.to(device)
    
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
        # for i in range(X.shape[0]):
        #     plt.plot([0., X[i].dot(first)], [0., X[i].dot(second)], color='k', alpha=0.5)


        plt.show()
        plt.pause(1.)
    
    #Evaluate teacher SD
    teacher_Win, teacher_wout = get_teacher(args.d, args.target_neurons, args.ball_radius, args.seed, args.positive_weights, args.n_negative_weights)
    teacher_train_SD = f1_stein_discrepancy(teacher_Win, teacher_wout, X, args)
    teacher_val_SD = f1_stein_discrepancy(teacher_Win, teacher_wout, Xval, args)
    teacher_test_SD = f1_stein_discrepancy(teacher_Win, teacher_wout, Xte, args)

    print(teacher_wout)

    teacher_f, teacher_f_stuff = free_energy_sampling(teacher_Win, teacher_wout, n_uniform_samples=args.eval_unif_samples, reps=args.eval_reps, get_stuff=True)

    teacher_train_cea = cross_entropy_avgterm(teacher_Win, teacher_wout, X)
    teacher_val_cea = cross_entropy_avgterm(teacher_Win, teacher_wout, Xval)
    teacher_test_cea = cross_entropy_avgterm(teacher_Win, teacher_wout, Xte)

    teacher_train_ce = teacher_train_cea + teacher_f
    teacher_val_ce = teacher_val_cea + teacher_f
    teacher_test_ce = teacher_test_cea + teacher_f

    print(f'Teacher F1 SD: train {teacher_train_SD}, val {teacher_val_SD}, test {teacher_test_SD}')
    
    Win = torch.randn(args.d, args.neurons)
    
    if args.model == 'f2':
        wout = torch.zeros(1, args.neurons)
        max_cosine = teacher_Win.t().matmul(Win / torch.norm(Win, dim=0).unsqueeze(0)).abs().max()
    elif args.model == 'f1':
        wout = torch.randn(1, args.neurons)
        Win *= np.sqrt(args.alpha)
        wout = torch.randn(1, args.neurons) * np.sqrt(args.alpha)
        
    Win.to(device)
    wout.to(device)
    
    iters = []
    train_SD, val_SD, test_SD = [], [], []
    train_cross_entropy = []
    val_cross_entropy = []
    test_cross_entropy = []
    free_energy_stuff = []
    max_cosines = []
    Win_list = []
    wout_list = []
    #if args.task_id_rep is not None:
    train_kl_div = []
    val_kl_div = []
    test_kl_div = []
    kl_div_2 = []
    
    
    for i in range(args.niter):
        tstart = time.time()
        #reinitialize Whin, whout at every iteration
        Whin_full = torch.zeros(args.d, args.neurons*args.d)
        whout_full = torch.zeros(1, args.neurons*args.d)
        dim_values = np.zeros(args.d) 
        for dim in range(args.d):
            Whin = torch.randn(args.d, args.neurons)
            whout = torch.randn(1, args.neurons)
            Whin.div_(torch.sqrt(torch.norm(whout.data*torch.norm(Whin.data, p=2, dim=0).unsqueeze(0), p=1)))
            whout.div_(torch.sqrt(torch.norm(whout.data*torch.norm(Whin.data, p=2, dim=0).unsqueeze(0), p=1)))
            for t in range(args.niter_stein_training):
                lr = decreasing_stepsize(args.lr_stein, args.lrb_stein, t)
                expected_stein = lambda Whin, whout: expectation_tr_stein_operator_comp(Win.t(),wout.t(),
                                                                                        Whin.t(),whout.t(),X,dim)
                Whin.requires_grad_()
                whout.requires_grad_()
                fun_value = expected_stein(Whin, whout)[0]
                grad = torch.autograd.grad(fun_value, [Whin, whout])
                gradhin = grad[0]
                gradhout = grad[1]
                Whin.detach_()
                whout.detach_()
                Whin.add_(lr * gradhin.data) 
                whout.add_(lr * gradhout.data) 
                Whin.div_(torch.sqrt(torch.norm(whout.data*torch.norm(Whin.data, p=2, dim=0).unsqueeze(0), p=1)))
                whout.div_(torch.sqrt(torch.norm(whout.data*torch.norm(Whin.data, p=2, dim=0).unsqueeze(0), p=1)))
            dim_values[dim] = expectation_tr_stein_operator_comp(Win.t(),wout.t(),Whin.t(),whout.t(),X,dim)[0]
            Whin_full[:,(args.neurons*dim):(args.neurons*(dim+1))] = Whin
            whout_full[:,(args.neurons*dim):(args.neurons*(dim+1))] = whout
            print("dimension", dim, "done", time.time()-tstart, dim_values[dim])
        print("h update done", time.time()-tstart)
        lr = decreasing_stepsize(args.lr, args.lrb, i)
        if args.model == 'f1':
            gradin = torch.zeros(args.d, args.neurons)
            gradout = torch.zeros(1, args.neurons)
            for dim in range(args.d):
                Whin = Whin_full[:,(args.neurons*dim):(args.neurons*(dim+1))]
                whout = whout_full[:,(args.neurons*dim):(args.neurons*(dim+1))]
                expected_stein_2 = lambda Win, wout: expectation_tr_stein_operator_comp(Win.t(),wout.t(),
                                                                                    Whin.t(),whout.t(),X,dim)
                Win.requires_grad_()
                wout.requires_grad_()
                fun_value = expected_stein_2(Win, wout)[0]
                grad = torch.autograd.grad(fun_value, [Win, wout])
                gradin += grad[0]*dim_values[dim]
                gradout += grad[1]*dim_values[dim]
                Win.detach_()
                wout.detach_()
            Win.sub_(25 * lr * args.neurons * (gradin.data + args.wd * Win.data)) 
            wout.sub_(lr * args.neurons * (gradout.data + args.wd * wout.data)) 
            max_cosine = teacher_Win.t().matmul(Win.data / torch.norm(Win.data, dim=0).unsqueeze(0)).abs().max()
        elif args.model == 'f2':
            gradout = torch.zeros(1, args.neurons)
            for dim in range(args.d):
                Whin = Whin_full[:,(args.neurons*dim):(args.neurons*(dim+1))]
                whout = whout_full[:,(args.neurons*dim):(args.neurons*(dim+1))]
                expected_stein_2 = lambda wout: expectation_tr_stein_operator_comp(Win.t(),wout.t(),Whin.t(),whout.t(),X,dim)
                wout.requires_grad_()
                fun_value = expected_stein_2(wout)[0]
                grad = torch.autograd.grad(fun_value, wout)[0]
                gradout += grad*dim_values[dim]
                wout.detach_()
            wout.sub_(lr * args.neurons * (gradout.data + args.wd * wout.data)) 
            
        print("Iteration", i, "done. Max cosine:", max_cosine, "Stein discrepancy:", np.linalg.norm(dim_values), time.time()-tstart)
        print("Output Weights grad norm:", torch.norm(gradout))
        if args.model == 'f1':
            print("Positions grad norm:", torch.norm(gradin), "F1 norm", torch.norm(torch.norm(Win, p=2, dim=0).unsqueeze(0)*wout, p=1)/args.neurons)
            
        
        if i % args.eval_delta == 0:
            train_stein = f1_stein_discrepancy(Win, wout, X, args)
            val_stein = f1_stein_discrepancy(Win, wout, Xval, args)
            test_stein = f1_stein_discrepancy(Win, wout, Xte, args)

            kl_2 = compute_kl_a_posteriori(teacher_Win, teacher_wout, Win, wout, n_uniform_samples=30000, reps=5)
            ce_f, ce_f_stuff = free_energy_sampling(Win, wout, n_uniform_samples=args.eval_unif_samples, reps=args.eval_reps, get_stuff=True)

            train_cea = cross_entropy_avgterm(Win, wout, X)
            val_cea = cross_entropy_avgterm(Win, wout, Xval)
            test_cea = cross_entropy_avgterm(Win, wout, Xte)

            train_ce = train_cea + ce_f
            val_ce = val_cea + ce_f
            test_ce = test_cea + ce_f
            
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
            print(f'F1 SD: train {train_stein:.6f} (teacher train: {teacher_train_SD:.6f}), val {val_stein:.6f} (teacher val: {teacher_val_SD:.6f}), test {test_stein:.6f} (teacher test: {teacher_test_SD:.6f}). Cosine max {max_cosine:.6f}. ({dt_tot:.2f})', flush=True)
            print(f'Cross entropy: train {train_ce:.6f} (teacher train: {teacher_train_ce:.6f}), val {val_ce:.6f} (teacher val: {teacher_val_ce:.6f}), test {test_ce:.6f} (teacher test: {teacher_test_ce:.6f}).', flush=True)

            iters.append(i)
            train_SD.append(train_stein)
            val_SD.append(val_stein)
            test_SD.append(test_stein)
            train_cross_entropy.append(train_ce)
            val_cross_entropy.append(val_ce)
            test_cross_entropy.append(test_ce)
            free_energy_stuff.append(ce_f_stuff)
            max_cosines.append(max_cosine)

            Win_list.append(Win)
            wout_list.append(wout)
            
            #if args.task_id_rep is not None:
            train_kl_div.append(train_kl)
            val_kl_div.append(val_kl)
            test_kl_div.append(test_kl)
            kl_div_2.append(kl_2)

            res = {
                'teacher_train_SD': teacher_train_SD,
                'teacher_val_SD': teacher_val_SD,
                'teacher_test_SD': teacher_test_SD,
                'train_SD': train_SD,
                'val_SD': val_SD,
                'test_SD': test_SD,
                'teacher_train_ce': teacher_train_ce,
                'teacher_val_ce': teacher_val_ce,
                'teacher_test_ce': teacher_test_ce,
                'train_kl_div': train_kl_div,
                'val_kl_div': val_kl_div,
                'test_kl_div': test_kl_div,
                'kl_div_2': kl_div_2,
                'train_cross_entropy': train_cross_entropy,
                'val_cross_entropy': val_cross_entropy,
                'test_cross_entropy': test_cross_entropy,
                'teacher_f_stuff': teacher_f_stuff,
                'free_energy_stuff': free_energy_stuff,
                'max_cosines': max_cosines,
                'iters': iters,
                'teacher_Win': teacher_Win,
                'teacher_wout': teacher_wout,
                'Win': Win_list,
                'wout': wout_list
            }
                
            if not args.interactive:
                pickle.dump(res, open(fname, 'wb'))
