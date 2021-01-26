#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 10:17:53 2021

@author: tomachache
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import special

from qiskit import *
from qiskit.quantum_info.states.measures import state_fidelity

from QNN import QNN
from State_preparation import state_preparation
from Stack_QAE import stacked_QAE_fidelity


# Compute mean fidelity for multiple pairs
def fidelity(L):
    # L : list of input/target states
    
    backend = Aer.get_backend('statevector_simulator')
    fid = []
    for pair in L:
        state_0 = execute(pair[0], backend).result().get_statevector(pair[0])
        state_1 = execute(pair[1], backend).result().get_statevector(pair[1])
        fid.append(state_fidelity(state_0, state_1))
    return np.array(fid)


# Theoretical fidelity of noise-corrupted m-qubit GHZ states with GHZ
def theoretical_fid(m, p, noise_type = 'bitflip'):
    # m : nb of qubits
    # p : noise strength
    # noise_type : type of noise (bitflip, QDC)
    
    if noise_type == 'bitflip':
        return (1-p)**m + p**m
    elif noise_type == 'QDC': # Quantum Depolarizing Channel
        return 2**(m-1) * (p/4)**m + np.sum([special.binom(m, k) * (p/4)**k * (1-3*p/4)**(m-k) for k in range(m+1) if k%2 == 0], axis=0)
    else:
        raise ValueError('Unrecognized noise type.')
        
        
# Compute noisy/denoised fidelity for various noise strength, and when stacking QAE twice
def compute_robustness(QAE, P, nb_test_states, noise_name = 'noisy_GHZ_bitflip', stack = 1):
    # QAE : QAE to use
    # P : noise range
    # nb_test_states : nb of test states
    # noise_name : name of noise used in state preparation
    # stack : nb of times we stack the QAE
    
    fid_orig = []
    fid_mean = []
    fid_std = []
    fid2_mean = []
    fid2_std = []
    for p in P:
        test_states = [(state_preparation(QAE.M[0], noise_name, p), state_preparation(QAE.M[0], 'GHZ', 0)) for _ in range(nb_test_states)]
        fid_orig.append(np.mean(fidelity(test_states)))
        fid = QAE.test(test_states)
        fid_mean.append(np.mean(fid))
        fid_std.append(np.std(fid))
        
        fid2 = stacked_QAE_fidelity(QAE, test_states, stack)
        fid2_mean.append(np.mean(fid2))
        fid2_std.append(np.std(fid2))

    return fid_orig, fid_mean, fid_std, fid2_mean, fid2_std


# Plot mean and deviation of fidelity of QAE (alone or stacked twice) over various noise strength
def plot_fid_VS_noise_strength(QAE, P, nb_test_states, noise_name = 'noisy_GHZ_QDC', noise_type = 'QDC', stack = 1, filename = None):
    # QAE : QAE to use
    # P : noise range
    # nb_test_states : nb of test states to use for each noise strength
    # noise_name : name of noise used in state preparation
    # noise_type : type of noise (bitflip, QDC) for theoretical fidelity
    # stack : nb of times we stack the QAE
    
    fid_orig, fid_mean, fid_std, fid2_mean, fid2_std = compute_robustness(QAE, P, nb_test_states, noise_name = noise_name, stack = stack)
    fid_theo = theoretical_fid(QAE.M[0], P, noise_type = noise_type)
    
    plt.plot(P, fid_theo, 'rx', label = 'Noisy theoretical fidelity')
    plt.plot(P, fid_orig, 'gx', label = 'Noisy actual fidelity')
    #plt.errorbar(P, fid_mean, fid_std, fmt = 'bo', capsize = 5, mfc='white', label = 'Denoised fidelity')
    plt.errorbar(P, fid2_mean, fid2_std, fmt = 'mo', capsize = 5, mfc='white', label = 'Denoised fidelity {}x'.format(stack))
    plt.grid(lw = 0.5, ls = 'dotted')
    plt.xlabel('Noise strength of QDC')
    #plt.xlabel('Bit-flip probability')
    plt.ylabel('Fidelity with GHZ')
    plt.legend()
    
    if filename != None:
        plt.savefig(filename + '.pdf', bbox_inches = 'tight')


if __name__ == "__main__":
    
    # Create QAE and load weights
    QAE = QNN([3,1,3])
    K = np.load('Saved_models/Depolarizing_channel/K-3,1,3. p = 0.2. 150 pairs, eps = 0.1. eta = 0.25. Depolarizing channel.npy')
    QAE.set_K(K)
    
    # Create test states
    nb_test_states = 200
    p = 0.5
    test_states = [(state_preparation(QAE.M[0], 'noisy_GHZ_QDC', p), state_preparation(QAE.M[0], 'GHZ', 0)) for _ in range(nb_test_states)]
    
    # Get fidelities
    orig_fid = fidelity(test_states) # original (noisy) fidelity
    fid = QAE.test(test_states) # list of denoised fidelities
    
    print('Fidelity of raw states : {:.2f} +/- {:.2f}'.format(np.mean(orig_fid), np.std(orig_fid)))
    print('Fidelity of denoised states : {:.2f} +/- {:.2f}'.format(np.mean(fid), np.std(fid)))
    
    # Use LaTeX font for plots
    plt.rc('text', usetex = True)
    plt.rc('font', family = 'serif')
    
    # Get mean and deviation VS noise strength plot
    P = np.linspace(0, 1, 11)
    stack = 2 # stack QAE twice
    plot_fid_VS_noise_strength(QAE, P, nb_test_states, stack = stack, filename = 'fid_depo')
