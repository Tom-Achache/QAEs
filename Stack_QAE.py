#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 11:05:28 2021

@author: tomachache
"""

import numpy as np
import matplotlib.pyplot as plt

from qiskit import *
from qiskit.quantum_info.states.measures import state_fidelity
from qiskit.quantum_info import partial_trace

from QNN import QNN
from State_preparation import state_preparation


# Compute fidelity when stacking QAE k times
def stacked_QAE_fidelity(QAE, L, k):
    # QAE : QAE to use
    # L : list of test states
    # k : stacking maximum
    
    backend = Aer.get_backend('statevector_simulator')
    
    QAE_circ, out_qubits = QAE.subroutine_2()
    circ_0 = QuantumCircuit(QAE.W)
    circ_0.append(L[0][0], range(QAE.M[0])) 
    
    for i in range(k):
        circ_0.append(QAE_circ, range(QAE.W))
        circ_0.reset(list(set(range(QAE.W)) - set(out_qubits)))
    
    fid = []
    for pair in L:
        circ_0.data[0] = (pair[0], circ_0.data[0][1], circ_0.data[0][2])
        state_0 = partial_trace(execute(circ_0, backend).result().get_statevector(), list(set(range(QAE.W)) - set(out_qubits)))
        state_1 = execute(pair[1], backend).result().get_statevector()
        fid.append(state_fidelity(state_0, state_1))
        
    return fid


# Plot list of fidelities when stacking QAE up to k times
def stacked_QAE_fidelity_range(QAE, L, k, filename = None):
    # QAE : QAE to use
    # L : list of test states
    # k : stacking maximum
    # filename : whether to save figure and figure name
    
    backend = Aer.get_backend('statevector_simulator')
    
    QAE_circ, out_qubits = QAE.subroutine_2()
    circ_0 = QuantumCircuit(QAE.W)
    circ_0.append(L[0][0], range(QAE.M[0]))    
    
    all_fid_mean = []
    all_fid_std = []
    for i in range(k):
        circ_0.append(QAE_circ, range(QAE.W))
        circ_0.reset(list(set(range(QAE.W)) - set(out_qubits)))
        
        fid = []
        for pair in L:
            circ_0.data[0] = (pair[0], circ_0.data[0][1], circ_0.data[0][2])
            state_0 = partial_trace(execute(circ_0, backend).result().get_statevector(), list(set(range(QAE.W)) - set(out_qubits)))
            state_1 = execute(pair[1], backend).result().get_statevector()
            fid.append(state_fidelity(state_0, state_1))
            
        all_fid_mean.append(np.mean(fid))
        all_fid_std.append(np.std(fid))
    
    # Plot
    plt.errorbar(range(1,k+1), all_fid_mean, all_fid_std, marker = 'D', ms = 4, color = '#0000CC', capsize = 5)
    plt.grid(lw = 0.5, ls = 'dotted')
    plt.xlabel('Number of QAEs stacked')
    plt.ylabel('Fidelity with GHZ')
    
    if filename != None:
        plt.savefig(filename + '.pdf', bbox_inches = 'tight')
        
        
if __name__ == "__main__":
    
    # Create QAE and load weights
    QAE = QNN([3,1,3])
    K = np.load('Saved_models/Depolarizing_channel/K-3,1,3. p = 0.2. 150 pairs, eps = 0.1. eta = 0.25. Depolarizing channel.npy')
    QAE.set_K(K)
    
    # Create test states
    nb_test_states = 200
    p = 0.4
    test_states = [(state_preparation(QAE.M[0], 'noisy_GHZ_QDC', p), state_preparation(QAE.M[0], 'GHZ', 0)) for _ in range(nb_test_states)]
    
    # Use LaTeX font for plots
    plt.rc('text', usetex = True)
    plt.rc('font', family = 'serif')
    
    # Get stacked QAEs fidelities plot
    stack_max = 15
    stacked_QAE_fidelity_range(QAE, test_states, stack_max, filename = 'stacked_QAEs')
    
