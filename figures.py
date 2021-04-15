import matplotlib
matplotlib.use('Agg')

import glob
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from mle import compute_kl_a_posteriori

plt.style.use('ggplot')

CLEAR_BEST = False #True
SAVE_BEST = False #True
removed_set = set()

def val_test(model='f2', d=10, n=100, verbose=False, prefix='mle',
             metric='cross_entropy', teacher_metric='ce', neurons=1, seed=42,
             positive_weights=None, n_negative_weights=None, get_stuff=False):
    if n_negative_weights is not None:
        name = f'res/{prefix}/{prefix}_{model}_{d}_{neurons}_{seed}_{n}_*_{positive_weights}_{n_negative_weights}.pkl'
    elif positive_weights is not None:
        name = f'res/{prefix}/{prefix}_{model}_{d}_{neurons}_{seed}_{n}_*_{positive_weights}.pkl'
    else:
        name = f'res/{prefix}/{prefix}_{model}_{d}_{neurons}_{seed}_{n}_*'
    val_xs = []
    test_xs = []
    fnames = glob.glob(name)
    eff_fnames = []
    stuffs = []
    assert len(fnames) > 0, 'no files! ({})'.format(name)
    for fname in fnames:
        res = pickle.load(open(fname, 'rb'))
        if metric != 'kl_div' and np.any(np.isnan(res[f'val_{metric}'])):
            if verbose:
                print('skipping nans...')
            continue
        if metric == 'kl_div' and np.any(np.isnan(res[f'kl_div_2'])):
            if verbose:
                print('skipping nans...')
            continue
        if metric == 'kl_div':
            idx = np.argmin(res[f'kl_div_2'])
            val_xs.append(res[f'kl_div_2'][idx])
            test_xs.append(res[f'kl_div_2'][idx])
        else:
            idx = np.argmin(res[f'val_{metric}'])
            val_xs.append(res[f'val_{metric}'][idx])
            test_xs.append(res[f'test_{metric}'][idx])
        if get_stuff:
            stuffs.append(res[f'free_energy_stuff'][idx])
        eff_fnames.append(fname)
        if verbose and metric != 'kl_div':
            print(fname, len(res[f'val_{metric}']), idx, res[f'val_{metric}'][idx], res[f'test_{metric}'][idx])
        if verbose and metric == 'kl_div':
            print(fname, len(res[f'kl_div_2']), idx, res[f'kl_div_2'][idx])

    idx = np.argmin(val_xs)
    if verbose:
        print('best:', eff_fnames[idx])

    if SAVE_BEST:
        with open(f'res/{prefix}/hyper.txt', 'a') as f:
            f.write(eff_fnames[idx] + '\n')

    if teacher_metric == 'KSD':
        teacher_test = res[f'teacher_test_{teacher_metric}'][0]
    elif teacher_metric == 'kl':
        teacher_test = 0
    else:
        teacher_test = res[f'teacher_test_{teacher_metric}']

    if get_stuff:
        return test_xs[idx], teacher_test, stuffs[idx]
    else:
        return test_xs[idx], teacher_test


def test_sd(prefix='f1sd', model='f2', d=10, n=100, verbose=False, neurons=1, seed=42, positive_weights=None, n_negative_weights=None, get_stuff=None):
    return val_test(model=model, d=d, n=n, verbose=verbose, prefix=prefix, neurons=neurons,
                    metric='SD', teacher_metric='SD', seed=seed, n_negative_weights=n_negative_weights,
                    positive_weights=positive_weights)


def test_xe(prefix='mle', model='f2', d=10, n=100, verbose=False, positive_weights=None, n_negative_weights=None, neurons=1, seed=42, get_stuff=False):
    return val_test(model=model, d=d, n=n, verbose=verbose, prefix=prefix, neurons=neurons,
                        metric='cross_entropy', teacher_metric='ce', seed=seed,
                        positive_weights=positive_weights, n_negative_weights=n_negative_weights, get_stuff=get_stuff)

def test_kl(prefix='kl', model='f2', d=10, n=100, verbose=False, positive_weights=None, n_negative_weights=None, neurons=1, seed=42, get_stuff=False):
    return val_test(model=model, d=d, n=n, verbose=verbose, prefix=prefix, neurons=neurons,
                        metric='kl_div', teacher_metric='kl', seed=seed,
                        positive_weights=positive_weights, n_negative_weights=n_negative_weights, get_stuff=get_stuff)

def test_ksd(prefix='ksd', model='f2', d=10, n=100, verbose=False, positive_weights=None, n_negative_weights=None, neurons=1, seed=42, get_stuff=None):
    return val_test(model=model, d=d, n=n, positive_weights=positive_weights, n_negative_weights=n_negative_weights, neurons=neurons, seed=seed,
                        verbose=verbose, prefix=prefix, metric='KSD', teacher_metric='KSD')

NS = [10, 30, 100, 300, 1000, 3000, 10000]

def plot_curves(prefix='mle', d=10, ns=None, neurons=1, seed=42, positive_weights=None, eval_fn=test_xe, n_negative_weights=None,
                train_metric='', test_metric='', F_confidence=False, show_teacher=True, teacher_val=None,
                show_teacher_err=False, show_reps=False, set_yscale=False, yscale_lower=0.1, yscale_upper=1):
    if CLEAR_BEST and prefix not in removed_set and os.path.exists(f'res/{prefix}/hyper.txt'):
        os.remove(f'res/{prefix}/hyper.txt')
        removed_set.add(prefix)

    print(prefix, d, test_metric)
    if ns is None:
        ns = NS
    f1eval = lambda n: eval_fn(prefix=prefix, model='f1', d=d, n=n, neurons=neurons, seed=seed,
                               positive_weights=positive_weights, n_negative_weights=n_negative_weights, get_stuff=F_confidence, verbose=True)
    f2eval = lambda n: eval_fn(prefix=prefix, model='f2', d=d, n=n, neurons=neurons, seed=seed,
                               positive_weights=positive_weights, n_negative_weights=n_negative_weights, get_stuff=F_confidence)

    def evalreps(n, model='f1'):
        vals = []
        teach = []
        for rep in range(10):
            # if rep == 5:
            #     continue
            try:
                v, t = eval_fn(prefix=prefix + f'_rep{rep}', model=model, d=d, n=n, neurons=neurons, seed=seed,
                    positive_weights=positive_weights, n_negative_weights=n_negative_weights, get_stuff=False)
            except Exception as e:
                print(e)
                continue
            vals.append(v)
            teach.append(t)
        return np.mean(vals), np.std(vals), np.mean(teach)


    if show_reps:
        f1data = list(zip(*(evalreps(n, 'f1') for n in ns)))
        f2data = list(zip(*(evalreps(n, 'f2') for n in ns)))
        f2_xs = np.array(f2data[0])
        f1_xs = np.array(f1data[0])
        f2_std = np.array(f2data[1])
        f1_std = np.array(f1data[1])
        teachers = np.array(f2data[2] + f1data[2])
    else:
        f2data = list(zip(*(f2eval(n) for n in ns)))
        f1data = list(zip(*(f1eval(n) for n in ns)))
        f2_xs = np.array(f2data[0])
        f1_xs = np.array(f1data[0])
        teachers = np.array(f2data[1] + f1data[1])
        if F_confidence:
            f2_std = np.array(list(np.std(s[0]) for s in f2data[2]))
            f1_std = np.array(list(np.std(s[0]) for s in f1data[2]))

    if test_metric != 'KL divergence':
        plt.semilogx(ns, f2_xs, label='f2')
        plt.semilogx(ns, f1_xs, label='f1')
        if show_reps or F_confidence: #to be changed after trying logs
            plt.fill_between(ns, (f2_xs - f2_std), (f2_xs + f2_std), alpha=.3)
            plt.fill_between(ns, (f1_xs - f1_std), (f1_xs + f1_std), alpha=.3)
    
    else:
        plt.loglog(ns, f2_xs, label='f2')
        plt.loglog(ns, f1_xs, label='f1')
        if show_reps or F_confidence: #to be changed after trying logs
            plt.fill_between(ns, (f2_xs - f2_std), (f2_xs + f2_std), alpha=.3)
            plt.fill_between(ns, (f1_xs - f1_std), (f1_xs + f1_std), alpha=.3)

    print(np.mean(teachers), teacher_val)
    if teacher_val is None:
        teacher_val = np.mean(teachers)
    if show_teacher:
        plt.hlines(teacher_val, 0, ns[-1], 'k', 'dashed', label='teacher mean')

        if show_teacher_err:
            plt.hlines(np.min(teachers), 0, ns[-1], 'k', 'dotted', label='teacher min')
            plt.hlines(np.max(teachers), 0, ns[-1], 'k', 'dotted', label='teacher max')

    plt.ylabel(f'test {test_metric}')
    plt.title(f'{train_metric} training (d = {d}, $w_1^* = 2$)')
    plt.xlabel('n (samples)')
    plt.legend()
    return teacher_val

if __name__ == '__main__':
    d = 15
    show_fconf = False
    fconf_suffix = '_conf' if show_fconf else ''
    show_reps = True
    reps_suffix = '_reps' if show_reps else ''

    if d == 15:
        CE_teacher_val = -0.034
    else:
        CE_teacher_val = None
    
    # MLE 1 neuron
    plt.figure(figsize=(4,3))
    ns = [100, 300, 1000, 3000, 10000, 30000]
    plot_curves(prefix='mle_oneneuron_norm2', d=d, ns=ns, seed=12, eval_fn=test_xe, train_metric='cross entropy', test_metric='cross entropy', F_confidence=show_fconf, show_reps=show_reps)
    plt.title(f'MLE training (d = {d}, $w_1^* = 2$)')
    plt.savefig(f'figures/mle2_1pos_{d}_evalce{fconf_suffix}{reps_suffix}.pdf', bbox_inches='tight', pad_inches=0)
    
    plt.figure(figsize=(4,3))
    ns = [100, 300, 1000, 3000, 10000, 30000]
    plot_curves(prefix='mle_oneneuron_norm2', d=d, ns=ns, seed=12, eval_fn=test_kl, train_metric='cross entropy', test_metric='KL divergence', F_confidence=show_fconf, show_reps=show_reps, show_teacher=False)
    plt.ylim(0.0003,10.2)
    plt.title(f'MLE training (d = {d}, $w_1^* = 2$)')
    plt.savefig(f'figures/mle2_1pos_{d}_evalkl{fconf_suffix}{reps_suffix}.pdf', bbox_inches='tight', pad_inches=0)
    
    plt.figure(figsize=(4,3))
    ns = [100, 300, 1000, 3000, 10000, 30000]
    plot_curves(prefix='mle_oneneuron_norm10', d=d, ns=ns, seed=12, eval_fn=test_xe, train_metric='cross entropy', test_metric='cross entropy', F_confidence=show_fconf, show_reps=show_reps)
    plt.title(f'MLE training (d = {d}, $w_1^* = 10$)')
    plt.savefig(f'figures/mle10_1pos_{d}_evalce{fconf_suffix}{reps_suffix}.pdf', bbox_inches='tight', pad_inches=0)
    
    plt.figure(figsize=(4,3))
    ns = [100, 300, 1000, 3000, 10000, 30000]
    plot_curves(prefix='mle_oneneuron_norm10', d=d, ns=ns, seed=12, eval_fn=test_kl, train_metric='cross entropy', test_metric='KL divergence', F_confidence=show_fconf, show_reps=show_reps, show_teacher=False)
    plt.title(f'MLE training (d = {d}, $w_1^* = 10$)')
    plt.savefig(f'figures/mle10_1pos_{d}_evalkl{fconf_suffix}{reps_suffix}.pdf', bbox_inches='tight', pad_inches=0)

    # KSD 1 neuron
    plt.figure(figsize=(4,3))
    ksd_ns = [100, 300, 1000, 3000, 10000]
    plot_curves(prefix='ksd_oneneuron_norm2', d=d, ns=ksd_ns, eval_fn=test_ksd, train_metric='KSD', test_metric='KSD', seed=12, show_reps=show_reps)
    plt.title(f'KSD training (d = {d}, $w_1^* = 2$)')
    plt.savefig(f'figures/ksd2_1pos_{d}_evalksd{reps_suffix}{reps_suffix}.pdf', bbox_inches='tight', pad_inches=0)

    plt.figure(figsize=(4,3))
    plot_curves(prefix='ksd_oneneuron_norm2', d=d, ns=ksd_ns, eval_fn=test_xe, train_metric='KSD', test_metric='cross entropy', seed=12, show_reps=show_reps)
    plt.title(f'KSD training (d = {d}, $w_1^* = 2$)')
    plt.savefig(f'figures/ksd2_1pos_{d}_evalce{reps_suffix}.pdf', bbox_inches='tight', pad_inches=0)

    plt.figure(figsize=(4,3))
    plot_curves(prefix='ksd_oneneuron_norm2', d=d, ns=ksd_ns, eval_fn=test_kl, train_metric='KSD', test_metric='KL divergence', seed=12, show_reps=show_reps, show_teacher=False)
    plt.ylim(0.0003,10.2)
    plt.title(f'KSD training (d = {d}, $w_1^* = 2$)')
    plt.savefig(f'figures/ksd2_1pos_{d}_evalkl{reps_suffix}.pdf', bbox_inches='tight', pad_inches=0)
    
    plt.figure(figsize=(4,3))
    ksd_ns = [100, 300, 1000, 3000, 10000]
    plot_curves(prefix='ksd_oneneuron_norm10', d=d, ns=ksd_ns, eval_fn=test_ksd, train_metric='KSD', test_metric='KSD', seed=12, show_reps=show_reps)
    plt.title(f'KSD training (d = {d}, $w_1^* = 10$)')
    plt.savefig(f'figures/ksd10_1pos_{d}_evalksd{reps_suffix}{reps_suffix}.pdf', bbox_inches='tight', pad_inches=0)

    plt.figure(figsize=(4,3))
    plot_curves(prefix='ksd_oneneuron_norm10', d=d, ns=ksd_ns, eval_fn=test_xe, train_metric='KSD', test_metric='cross entropy', seed=12, show_reps=show_reps)
    plt.title(f'KSD training (d = {d}, $w_1^* = 10$)')
    plt.savefig(f'figures/ksd10_1pos_{d}_evalce{reps_suffix}.pdf', bbox_inches='tight', pad_inches=0)

    plt.figure(figsize=(4,3))
    plot_curves(prefix='ksd_oneneuron_norm10', d=d, ns=ksd_ns, eval_fn=test_kl, train_metric='KSD', test_metric='KL divergence', seed=12, show_reps=show_reps, show_teacher=False)
    plt.ylim(0.005,0.4)
    plt.title(f'KSD training (d = {d}, $w_1^* = 10$)')
    plt.savefig(f'figures/ksd10_1pos_{d}_evalkl{reps_suffix}.pdf', bbox_inches='tight', pad_inches=0)

    # MLE 2 and 4 neurons
    plt.figure(figsize=(4,3))
    ns = [100, 300, 1000, 3000, 10000, 30000]
    plot_curves(prefix='twoneuron_norm5', d=d, ns=ns, seed=12, neurons=2, n_negative_weights=2, positive_weights=False, eval_fn=test_xe, train_metric='cross entropy', test_metric='cross entropy', F_confidence=show_fconf, show_reps=show_reps)
    plt.title(f'MLE training (d={d}, $w_i^* = -2.5$, J=2)')
    plt.savefig(f'figures/twomle5_{d}_evalce{fconf_suffix}{reps_suffix}.pdf', bbox_inches='tight', pad_inches=0)

    plt.figure(figsize=(4,3))
    ns = [30, 100, 300, 1000, 3000, 10000, 30000]
    plot_curves(prefix='twoneuron_norm5', d=d, ns=ns, seed=12, neurons=2, n_negative_weights=2, positive_weights=False, eval_fn=test_kl, train_metric='cross entropy', test_metric='KL divergence', F_confidence=show_fconf, show_reps=show_reps, show_teacher=False)
    plt.title(f'MLE training (d={d}, $w_i^* = -2.5$, J=2)')
    plt.savefig(f'figures/twomle5_{d}_evalkl{fconf_suffix}{reps_suffix}.pdf', bbox_inches='tight', pad_inches=0)
    
    plt.figure(figsize=(4,3))
    ns = [100, 300, 1000, 3000, 10000, 30000]
    plot_curves(prefix='twoneuron_norm10', d=d, ns=ns, seed=12, neurons=2, n_negative_weights=2, positive_weights=False, eval_fn=test_xe, train_metric='cross entropy', test_metric='cross entropy', F_confidence=show_fconf, show_reps=show_reps)
    plt.title(f'MLE training (d={d}, $w_i^* =-5$, J=2)')
    plt.savefig(f'figures/twomle10_{d}_evalce{fconf_suffix}{reps_suffix}.pdf', bbox_inches='tight', pad_inches=0)

    plt.figure(figsize=(4,3))
    ns = [100, 300, 1000, 3000, 10000, 30000]
    plot_curves(prefix='twoneuron_norm10', d=d, ns=ns, seed=12, neurons=2, n_negative_weights=2, positive_weights=False, eval_fn=test_kl, train_metric='cross entropy', test_metric='KL divergence', F_confidence=show_fconf, show_reps=show_reps, show_teacher=False)
    plt.ylim(0.001,2.5)
    plt.title(f'MLE training (d={d}, $w_i^* = -5$, J=2)')
    plt.savefig(f'figures/twomle10_{d}_evalkl{fconf_suffix}{reps_suffix}.pdf', bbox_inches='tight', pad_inches=0)
    
    plt.figure(figsize=(4,3))
    ns = [100, 300, 1000, 3000, 10000, 30000]
    plot_curves(prefix='fourneuron_norm30', d=d, ns=ns, seed=12, neurons=4, n_negative_weights=2, positive_weights=False, eval_fn=test_xe, train_metric='cross entropy', test_metric='cross entropy', F_confidence=show_fconf, show_reps=show_reps)
    plt.title(f'MLE training (d={d}, $w_i^* = \\pm 7.5$, J=4)')
    plt.savefig(f'figures/fourmle30_{d}_evalce{fconf_suffix}{reps_suffix}.pdf', bbox_inches='tight', pad_inches=0)

    plt.figure(figsize=(4,3))
    ns = [100, 300, 1000, 3000, 10000, 30000]
    plot_curves(prefix='fourneuron_norm30', d=d, ns=ns, seed=12, neurons=4, n_negative_weights=2, positive_weights=False, eval_fn=test_kl, train_metric='cross entropy', test_metric='KL divergence', F_confidence=show_fconf, show_reps=show_reps, show_teacher=False)
    plt.ylim(0.0035,0.35)
    plt.title(f'MLE training (d={d}, $w_i^* = \\pm 7.5$, J=4)')
    plt.savefig(f'figures/fourmle30_{d}_evalkl{fconf_suffix}{reps_suffix}.pdf', bbox_inches='tight', pad_inches=0)
    
    # KSD 2 and 4 neurons
    plt.figure(figsize=(4,3))
    ns = [100, 300, 1000, 3000, 10000]
    plot_curves(prefix='ksd_twoneuron_norm5', d=d, ns=ns, seed=12, neurons=2, n_negative_weights=2, positive_weights=False, eval_fn=test_xe, train_metric='KSD', test_metric='cross entropy', F_confidence=show_fconf, show_reps=show_reps)
    plt.title(f'KSD training (d={d}, $w_i^* = -2.5$, J=2)')
    plt.savefig(f'figures/twoksd5_{d}_evalce{fconf_suffix}{reps_suffix}.pdf', bbox_inches='tight', pad_inches=0)

    plt.figure(figsize=(4,3))
    ns = [100, 300, 1000, 3000, 10000]
    plot_curves(prefix='ksd_twoneuron_norm5', d=d, ns=ns, seed=12, neurons=2, n_negative_weights=2, positive_weights=False, eval_fn=test_kl, train_metric='KSD', test_metric='KL divergence', F_confidence=show_fconf, show_reps=show_reps, show_teacher=False)
    plt.ylim(0.002,4)
    plt.title(f'KSD training (d={d}, $w_i^* = -2.5$, J=2)')
    plt.savefig(f'figures/twoksd5_{d}_evalkl{fconf_suffix}{reps_suffix}.pdf', bbox_inches='tight', pad_inches=0)

    plt.figure(figsize=(4,3))
    ns = [100, 300, 1000, 3000, 10000]
    plot_curves(prefix='ksd_twoneuron_norm5', d=d, ns=ns, seed=12, neurons=2, n_negative_weights=2, positive_weights=False, eval_fn=test_ksd, train_metric='KSD', test_metric='KSD', F_confidence=show_fconf, show_reps=show_reps)
    plt.title(f'KSD training (d={d}, $w_i^* = -2.5$, J=2)')
    plt.savefig(f'figures/twoksd5_{d}_evalksd{fconf_suffix}{reps_suffix}.pdf', bbox_inches='tight', pad_inches=0)
        
    plt.figure(figsize=(4,3))
    ns = [100, 300, 1000, 3000, 10000]
    plot_curves(prefix='ksd_twoneuron_norm10', d=d, ns=ns, seed=12, neurons=2, n_negative_weights=2, positive_weights=False, eval_fn=test_xe, train_metric='KSD', test_metric='cross entropy', F_confidence=show_fconf, show_reps=show_reps)
    plt.title(f'KSD training (d={d}, $w_i^* = -5$, J=2)')
    plt.savefig(f'figures/twoksd10_{d}_evalce{fconf_suffix}{reps_suffix}.pdf', bbox_inches='tight', pad_inches=0)

    plt.figure(figsize=(4,3))
    ns = [100, 300, 1000, 3000, 10000]
    plot_curves(prefix='ksd_twoneuron_norm10', d=d, ns=ns, seed=12, neurons=2, n_negative_weights=2, positive_weights=False, eval_fn=test_kl, train_metric='KSD', test_metric='KL divergence', F_confidence=show_fconf, show_reps=show_reps, show_teacher=False)
    plt.ylim(0.001,2.5)
    plt.title(f'KSD training (d={d}, $w_i^* = -5$, J=2)')
    plt.savefig(f'figures/twoksd10_{d}_evalkl{fconf_suffix}{reps_suffix}.pdf', bbox_inches='tight', pad_inches=0)

    plt.figure(figsize=(4,3))
    ns = [100, 300, 1000, 3000, 10000]
    plot_curves(prefix='ksd_twoneuron_norm10', d=d, ns=ns, seed=12, neurons=2, n_negative_weights=2, positive_weights=False, eval_fn=test_xe, train_metric='KSD', test_metric='KSD', F_confidence=show_fconf, show_reps=show_reps)
    plt.title(f'KSD training (d={d}, $w_i^* = -5$, J=2)')
    plt.savefig(f'figures/twoksd10_{d}_evalKSD{fconf_suffix}{reps_suffix}.pdf', bbox_inches='tight', pad_inches=0)
    
    plt.figure(figsize=(4,3))
    ns = [100, 300, 1000, 3000, 10000]
    plot_curves(prefix='ksd_fourneuron_norm30', d=d, ns=ns, seed=12, neurons=4, n_negative_weights=2, positive_weights=False, eval_fn=test_xe, train_metric='KSD', test_metric='cross entropy', F_confidence=show_fconf, show_reps=show_reps)
    plt.title(f'KSD training (d={d}, $w_i^* = \\pm 7.5$, J=4)')
    plt.savefig(f'figures/fourksd30_{d}_evalce{fconf_suffix}{reps_suffix}.pdf', bbox_inches='tight', pad_inches=0)

    plt.figure(figsize=(4,3))
    ns = [100, 300, 1000, 3000, 10000]
    plot_curves(prefix='ksd_fourneuron_norm30', d=d, ns=ns, seed=12, neurons=4, n_negative_weights=2, positive_weights=False, eval_fn=test_kl, train_metric='KSD', test_metric='KL divergence', F_confidence=show_fconf, show_reps=show_reps, show_teacher=False)
    plt.ylim(0.0035,0.35)
    plt.title(f'KSD training (d={d}, $w_i^* = \\pm 7.5$, J=4)')
    plt.savefig(f'figures/fourksd30_{d}_evalkl{fconf_suffix}{reps_suffix}.pdf', bbox_inches='tight', pad_inches=0)

    plt.figure(figsize=(4,3))
    ns = [100, 300, 1000, 3000, 10000]
    plot_curves(prefix='ksd_fourneuron_norm30', d=d, ns=ns, seed=12, neurons=4, n_negative_weights=2, positive_weights=False, eval_fn=test_xe, train_metric='KSD', test_metric='KSD', F_confidence=show_fconf, show_reps=show_reps)
    plt.title(f'KSD training (d={d}, $w_i^* = \\pm 7.5$, J=4)')
    plt.savefig(f'figures/fourksd30_{d}_evalKSD{fconf_suffix}{reps_suffix}.pdf', bbox_inches='tight', pad_inches=0)
    
    # F1SD
    d = 10
    
    plt.figure(figsize=(4,3))
    ns = [100, 300, 1000, 3000]
    plot_curves(prefix='f1sdloop_oneneuron_norm2', d=d, ns=ns, seed=12, eval_fn=test_xe, train_metric='F1-SD', test_metric='cross entropy', F_confidence=show_fconf, show_reps=show_reps)
    plt.title(f'F1-SD training (d={d}, $w_1^* = 2$)')
    plt.savefig(f'figures/f1sd2_1pos_{d}_evalce{fconf_suffix}{reps_suffix}.pdf', bbox_inches='tight', pad_inches=0)
    
    plt.figure(figsize=(4,3))
    ns = [100, 300, 1000, 3000]
    plot_curves(prefix='f1sdloop_oneneuron_norm2', d=d, ns=ns, seed=12, eval_fn=test_kl, train_metric='F1-SD', test_metric='KL divergence', F_confidence=show_fconf, show_reps=show_reps, show_teacher=False)
    plt.ylim(0.0003,10.2)
    plt.title(f'F1-SD training (d={d}, $w_1^* = 2$)')
    plt.savefig(f'figures/f1sd2_1pos_{d}_evalkl{fconf_suffix}{reps_suffix}.pdf', bbox_inches='tight', pad_inches=0)

    plt.figure(figsize=(4,3))
    ns = [100, 300, 1000, 3000]
    plot_curves(prefix='f1sdloop_oneneuron_norm2', d=d, ns=ns, seed=12, eval_fn=test_sd, train_metric='F1-SD', test_metric='F1-SD', F_confidence=show_fconf, show_reps=show_reps)
    plt.title(f'F1-SD training (d={d}, $w_1^* = 2$)')
    plt.savefig(f'figures/f1sd2_1pos_{d}_evalf1sd{fconf_suffix}{reps_suffix}.pdf', bbox_inches='tight', pad_inches=0)
    
    plt.figure(figsize=(4,3))
    ns = [100, 300, 1000, 3000]
    plot_curves(prefix='f1sdloop_oneneuron_norm10', d=d, ns=ns, seed=12, eval_fn=test_xe, train_metric='F1-SD', test_metric='cross entropy', F_confidence=show_fconf, show_reps=show_reps)
    plt.title(f'F1-SD training (d={d}, $w_1^* = 10$)')
    plt.savefig(f'figures/f1sd10_1pos_{d}_evalce{fconf_suffix}{reps_suffix}.pdf', bbox_inches='tight', pad_inches=0)

    plt.figure(figsize=(4,3))
    ns = [100, 300, 1000, 3000]
    plot_curves(prefix='f1sdloop_oneneuron_norm10', d=d, ns=ns, seed=12, eval_fn=test_kl, train_metric='F1-SD', test_metric='KL divergence', F_confidence=show_fconf, show_reps=show_reps, show_teacher=False)
    plt.ylim(0.005,0.4)
    plt.title(f'F1-SD training (d={d}, $w_1^* = 10$)')
    plt.savefig(f'figures/f1sd10_1pos_{d}_evalkl{fconf_suffix}{reps_suffix}.pdf', bbox_inches='tight', pad_inches=0)

    plt.figure(figsize=(4,3))
    ns = [100, 300, 1000, 3000]
    plot_curves(prefix='f1sdloop_oneneuron_norm10', d=d, ns=ns, seed=12, eval_fn=test_sd, train_metric='F1-SD', test_metric='F1-SD', F_confidence=show_fconf, show_reps=show_reps)
    plt.title(f'F1-SD training (d={d}, $w_1^* = 10$)')
    plt.savefig(f'figures/f1sd10_1pos_{d}_evalf1sd{fconf_suffix}{reps_suffix}.pdf', bbox_inches='tight', pad_inches=0)

    plt.figure(figsize=(4,3))
    ns = [100, 300, 1000, 3000]
    plot_curves(prefix='f1sdloop_twoneuron_norm5', d=d, ns=ns, seed=12, neurons=2, n_negative_weights=2, positive_weights=False, eval_fn=test_xe, train_metric='F1-SD', test_metric='cross entropy', F_confidence=show_fconf, show_reps=show_reps)
    plt.title(f'F1-SD training (d={d}, $w_i^* = -2.5$, J=2)')
    plt.savefig(f'figures/twof1sd5_{d}_evalce{fconf_suffix}{reps_suffix}.pdf', bbox_inches='tight', pad_inches=0)

    plt.figure(figsize=(4,3))
    ns = [100, 300, 1000, 3000]
    plot_curves(prefix='f1sdloop_twoneuron_norm5', d=d, ns=ns, seed=12, neurons=2, n_negative_weights=2, positive_weights=False, eval_fn=test_kl, train_metric='F1-SD', test_metric='KL divergence', F_confidence=show_fconf, show_reps=show_reps, show_teacher=False)
    plt.ylim(0.002,4)
    plt.title(f'F1-SD training (d={d}, $w_i^* = -2.5$, J=2)')
    plt.savefig(f'figures/twof1sd5_{d}_evalkl{fconf_suffix}{reps_suffix}.pdf', bbox_inches='tight', pad_inches=0)

    plt.figure(figsize=(4,3))
    ns = [10, 30, 100, 300, 1000, 3000]
    plot_curves(prefix='f1sdloop_twoneuron_norm5', d=d, ns=ns, seed=12, neurons=2, n_negative_weights=2, positive_weights=False, eval_fn=test_sd, train_metric='F1-SD', test_metric='F1-SD', F_confidence=show_fconf, show_reps=show_reps)
    plt.title(f'F1-SD training (d={d}, $w_i^* = -2.5$, J=2)')
    plt.savefig(f'figures/twof1sd5_{d}_evalf1sd{fconf_suffix}{reps_suffix}.pdf', bbox_inches='tight', pad_inches=0)
    
    plt.figure(figsize=(4,3))
    ns = [100, 300, 1000, 3000]
    plot_curves(prefix='f1sdloop_twoneuron_norm10', d=d, ns=ns, seed=12, neurons=2, n_negative_weights=2, positive_weights=False, eval_fn=test_xe, train_metric='F1-SD', test_metric='cross entropy', F_confidence=show_fconf, show_reps=show_reps)
    plt.title(f'F1-SD training (d={d}, $w_i^* = -5$, J=2)')
    plt.savefig(f'figures/twof1sd10_{d}_evalce{fconf_suffix}{reps_suffix}.pdf', bbox_inches='tight', pad_inches=0)

    plt.figure(figsize=(4,3))
    ns = [100, 300, 1000, 3000]
    plot_curves(prefix='f1sdloop_twoneuron_norm10', d=d, ns=ns, seed=12, neurons=2, n_negative_weights=2, positive_weights=False, eval_fn=test_kl, train_metric='F1-SD', test_metric='KL divergence', F_confidence=show_fconf, show_reps=show_reps, show_teacher=False)
    plt.ylim(0.001,2.5)
    plt.title(f'F1-SD training (d={d}, $w_i^* = -5$, J=2)')
    plt.savefig(f'figures/twof1sd10_{d}_evalkl{fconf_suffix}{reps_suffix}.pdf', bbox_inches='tight', pad_inches=0)

    plt.figure(figsize=(4,3))
    ns = [100, 300, 1000, 3000]
    plot_curves(prefix='f1sdloop_twoneuron_norm10', d=d, ns=ns, seed=12, neurons=2, n_negative_weights=2, positive_weights=False, eval_fn=test_sd, train_metric='F1-SD', test_metric='F1-SD', F_confidence=show_fconf, show_reps=show_reps)
    plt.title(f'F1-SD training (d={d}, $w_i^* = -5$, J=2)')
    plt.savefig(f'figures/twof1sd10_{d}_evalf1sd{fconf_suffix}{reps_suffix}.pdf', bbox_inches='tight', pad_inches=0)
    
    plt.figure(figsize=(4,3))
    ns = [100, 300, 1000, 3000]
    plot_curves(prefix='f1sdloop_fourneuron_norm30', d=d, ns=ns, seed=12, neurons=4, n_negative_weights=2, positive_weights=False, eval_fn=test_xe, train_metric='F1-SD', test_metric='cross entropy', F_confidence=show_fconf, show_reps=show_reps)
    plt.title(f'F1-SD training (d={d}, $w_i^* = \\pm 7.5$, J=4)')
    plt.savefig(f'figures/fourf1sd30_{d}_evalce{fconf_suffix}{reps_suffix}.pdf', bbox_inches='tight', pad_inches=0)

    plt.figure(figsize=(4,3))
    ns = [100, 300, 1000, 3000]
    plot_curves(prefix='f1sdloop_fourneuron_norm30', d=d, ns=ns, seed=12, neurons=4, n_negative_weights=2, positive_weights=False, eval_fn=test_kl, train_metric='F1-SD', test_metric='KL divergence', F_confidence=show_fconf, show_reps=show_reps, show_teacher=False)
    plt.ylim(0.0035,0.35)
    plt.title(f'F1-SD training (d={d}, $w_i^* = \\pm 7.5$, J=4)')
    plt.savefig(f'figures/fourf1sd30_{d}_evalkl{fconf_suffix}{reps_suffix}.pdf', bbox_inches='tight', pad_inches=0)

    plt.figure(figsize=(4,3))
    ns = [100, 300, 1000, 3000]
    plot_curves(prefix='f1sdloop_fourneuron_norm30', d=d, ns=ns, seed=12, neurons=4, n_negative_weights=2, positive_weights=False, eval_fn=test_sd, train_metric='F1-SD', test_metric='F1-SD', F_confidence=show_fconf, show_reps=show_reps)
    plt.title(f'F1-SD training (d={d}, $w_i^* = \\pm 7.5$, J=4)')
    plt.savefig(f'figures/fourf1sd30_{d}_evalf1sd{fconf_suffix}{reps_suffix}.pdf', bbox_inches='tight', pad_inches=0)
     
