#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 10:23:22 2021

@author: tomachache
"""

import numpy as np
import matplotlib.pyplot as plt

from QNN import QNN
from State_preparation import state_preparation
from test import fidelity

# Plot noise robustness of gates (introduce noise in the QNN's unitary matrices)
def noise_robustness(QNN, K, test_states, G, filename = None):
    # QNN : QNN to use
    # K : vector of coefficients for the QNN's matrices
    # test_states : test states to consider
    # G : noise range
    # filename : whether to save figure and figure name

    mean = []
    std = []
    
    for g in G:
        noise = np.random.normal(0, g, len(K))
        QNN.set_K(K + noise)
        fid = QNN.test(test_states)
        mean.append(np.mean(fid))
        std.append(np.std(fid))
    
    orig_fid = np.mean(fidelity(test_states)) # original (noisy) fidelity
    
    plt.errorbar(G, mean, std, fmt = 'bo', capsize = 5, mfc = 'white', label = 'Denoised fidelity')
    plt.grid(lw = 0.5, ls = 'dotted')
    plt.axhline(orig_fid, color = 'black', linestyle = '--', linewidth = 0.7, label = 'Mean original fidelity')
    plt.xlabel('Standard deviation')
    plt.ylabel('Fidelity with GHZ')
    #plt.title("[3,1,3] QAE. QDC with noise strength $p = 0.3$")
    plt.legend()
    
    if filename != None:
        plt.savefig(filename + '.pdf', bbox_inches = 'tight')


if __name__ == "__main__":
    
    # Create QAE and load weights
    QAE = QNN([3,1,3])
    K = np.load('/Users/tomachache/Downloads/Columbia/Spring 2020/Quantum Computing - Theory and Practice/Project/Saved_model/Depolarizing_channel/K-3,1,3. p = 0.2. 150 pairs, eps = 0.1. eta = 0.25. Depolarizing channel.npy')
    QAE.set_K(K)
    
    # Print coefficients K
    print('K : {:.2f} +/- {:.2f}'.format(np.mean(K), np.std(K)))
    
    # Pick noise range accordingly
    G = np.linspace(0, 0.2, 21)
    
    # Create test states
    nb_test_states = 200
    p = 0.3
    test_states = [(state_preparation(QAE.M[0], 'noisy_GHZ_QDC', p), state_preparation(QAE.M[0], 'GHZ', 0)) for _ in range(nb_test_states)]
    
    # Use LaTeX font for plots
    plt.rc('text', usetex = True)
    plt.rc('font', family = 'serif')
    
    # Get robustness plot
    noise_robustness(QAE, K, test_states, G, filename = 'noise_robustness')