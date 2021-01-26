#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 11:03:21 2021

@author: tomachache
"""

import numpy as np
import matplotlib.pyplot as plt

from qiskit import *
from qiskit.quantum_info import partial_trace
from qiskit.visualization import plot_state_city, plot_state_hinton

from QNN import QNN
from State_preparation import state_preparation


# Get average statevector of noisy/denoised states
def get_statevectors(QAE, noisy_states):
    # QAE : QAE to use
    # noisy_states : list of noisy states
    
    backend = Aer.get_backend('statevector_simulator')
    QAE_circ, out_qubits = QAE.subroutine_2()
    
    state_0 = np.zeros((2**QAE.M[0],2**QAE.M[0]), dtype = 'complex128')
    state_1 = np.zeros(2**QAE.M[0], dtype = 'complex128')
    for state in noisy_states:
        circ_0 = QuantumCircuit(QAE.W)
        circ_0.append(state, range(QAE.M[0]))
        circ_0.append(QAE_circ, range(QAE.W))
        
        state_0 += partial_trace(execute(circ_0, backend).result().get_statevector(), list(set(range(QAE.W)) - set(out_qubits))).data
        state_1 += execute(state, backend).result().get_statevector()
    return (state_0/len(noisy_states), state_1/len(noisy_states))


# Plot State City of noisy and denoised states
def plot_SC(state_0, state_1, filename = None):
    # state_0 : statevector of denoised state
    # state_1 : statevector of noisy state
    # filename : whether to save figure and figure name
    
    fig = plt.figure(figsize=(15,10))
    plt.title("States city of noisy (top) and denoised (bottom) states. We used a [2,1,2] QAE, and 200 GHZ states affected by the QDC with noise strength $p = 0.4$. \n The states city are respectively the average initial state vector and the average density matrix output by the QAE.")
    plt.axis("off")
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    plot_state_city(state_1, color = ['crimson', 'crimson'], ax_real = ax1, ax_imag = ax2)
    plot_state_city(state_0, color = ['crimson', 'crimson'], ax_real = ax3, ax_imag = ax4)
    ax1.set_zlim(0,0.5)
    ax2.set_zlim(0,0.5)
    ax3.set_zlim(0,0.5)
    ax4.set_zlim(0,0.5)
    
    if filename != None:
        plt.savefig(filename + '.pdf', bbox_inches = 'tight')
        

# Plot Hinton diagram of noisy and denoised states
def plot_SH(state_0, state_1, filename = None):
    # state_0 : statevector of denoised state
    # state_1 : statevector of noisy state
    # filename : whether to save figure and figure name
    
    fig = plt.figure(figsize=(15,10))
    plt.axis("off")
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)
    plot_state_hinton(state_1, ax_real = ax1, ax_imag = ax2)
    plot_state_hinton(state_0, ax_real = ax3, ax_imag = ax4)
    
    if filename != None:
        plt.savefig(filename + '.pdf', bbox_inches = 'tight')


if __name__ == "__main__":
    
    # Create QAE and load weights
    QAE = QNN([2,1,2])
    K = np.load('/Users/tomachache/Downloads/Columbia/Spring 2020/Quantum Computing - Theory and Practice/Project/Saved_model/Depolarizing_channel/K-2,1,2. p = 0.2. 100 pairs, eps = 0.1. eta = 0.25. Depolarizing channel.npy')
    QAE.set_K(K)
    
    # Create noisy states
    nb_states = 200
    p = 0.4
    noisy_states = [state_preparation(QAE.M[0], 'noisy_GHZ_QDC', p) for _ in range(nb_states)]
    
    # Get statevectors
    state_0, state_1 = get_statevectors(QAE, noisy_states)
    
    # Use LaTeX font for plots
    plt.rc('text', usetex = True)
    plt.rc('font', family = 'serif')
    
    # Get State City
    plot_SC(state_0, state_1, filename = 'State City')
    
    # Get Hinton diagram
    plot_SH(state_0, state_1, filename = 'Hinton diagram')
    