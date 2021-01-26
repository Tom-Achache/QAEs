#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 18:13:18 2021

@author: tomachache
"""

from qiskit import *
import numpy as np
import matplotlib.pyplot as plt
from QNN import QNN


# Create QDC circuit
def QDC(m, p):
    # m : nb of qubits
    # p : noise strength of QDC
    
    circ = QuantumCircuit(m, name = 'Depolarizing Channel')
    
    probas = [1 - 3*p/4, p/4, p/4, p/4]
    gate_inds = np.random.choice(np.arange(4), size = m, p = probas)
    
    for k in range(m):
        if gate_inds[k] == 1:
            circ.x(k)
        elif gate_inds[k] == 2:
            circ.y(k)
        elif gate_inds[k] == 3:
            circ.z(k)
    return circ


# Create noisy QSS circuit
def noisy_QSS_circuit(m = 3, p = 0.2, QAE = None):
    # m : nb of qubits
    # p : noise strength of QDC
    # QAE : specify whether to denoise and the Quantum Autoencoder to use
    
    R = range(m)
    
    if QAE is not None:
        circ = QuantumCircuit(QAE.W, m, name = 'QSS')
    else:
        circ = QuantumCircuit(m, m, name = 'QSS')
        
    # Initialize the GHZ triplet
    circ.h(0)
    for k in R[1:]:
        circ.cx(0,k)
    
    # Pass it through QDC
    circ.barrier()
    circ.append(QDC(m, p).to_instruction(), R)
    circ.barrier()
    
    if QAE is not None: # use QAE to denoise
        QAE_circ, out_qubits = QAE.subroutine_2()
        R = list(out_qubits)
        QAE_circ.name = 'Quantum Autoencoder'
        circ.append(QAE_circ, range(QAE.W))
        circ.barrier()
    
    # Each participant make a measurement according to the X or Y basis at random
    p = np.random.rand(m+1)
    basis = '' # will record the basis we chose for measurement
    for k in R:
        if p[k] < 1/2: # make a measurement in the X basis
            basis += 'X'
            circ.h(k)
        else: # make a measurement in the Y basis
            basis += 'Y'
            circ.sdg(k)
            circ.h(k)
    circ.barrier()
    
    circ.measure(R, R)
    return circ, basis


# Simulate noisy QSS protocol
def noisy_QSS_protocol(nb_trials, m = 3, p = 0.2, QAE = None):
    # nb_trials : nb of times we simulate protocol
    # m : nb of qubits
    # p : noise strength of QDC
    # QAE : specify whether to denoise and the Quantum Autoencoder to use
    
    Valid_basis = ['XXX', 'YYX', 'XYY', 'YXY']
    
    key1 = '' # final key of Charlie
    key2 = '' # final key of Alice and Bob
    
    for _ in range(nb_trials):
        circ, basis = noisy_QSS_circuit(m = m, p = p, QAE = QAE) # create new noisy QSS circuit
    
        if basis not in Valid_basis:
            #print('Invalid basis. Too bad !')
            continue
        else: # Alice, Bob and Charlie are supposed to be able to share a qubit
            result = execute(circ, Aer.get_backend('qasm_simulator'), shots = 1).result()
            res = next(iter(result.get_counts())) # get first (and only) key --> resulting bits
            Alice = int(res[0])
            Bob = int(res[1])
            Charlie = int(res[2])
            
            if basis == Valid_basis[0]: # XXX
                key1 += str(Charlie)
                key2 += str((Alice + Bob) % 2)
            else: # YYX, XYY or YXY
                key1 += str(Charlie)
                key2 += str((Alice + Bob + 1) % 2)
    #print('Found secret key of length {}.'.format(len(key1)))
    return key1, key2


# Compute normalized difference between two keys (--> failure probability of QSS protocol)
def p_diff(key1, key2):
    # key1, key2 : strings of bits
    
    assert len(key1) == len(key2), "Lengths of keys don't match"
    return np.sum(np.array(list(key1)) != np.array(list(key2)))/len(key1)


# Compute failure probability of noisy/denoised QSS protocol for various strength of QDC
def noise_strength_VS_QSS(nb_trials, m = 3, QAE = None, P = np.linspace(0,1,11)):
    # nb_trials : nb of times we simulate protocol
    # m : nb of qubits
    # QAE : specify whether to denoise and the Quantum Autoencoder to use
    # P : list of QDC strengths
    
    list_diff = []
    for p in P:
        key1, key2 = noisy_QSS_protocol(nb_trials = nb_trials, m = m, p = p, QAE = QAE)
        list_diff.append(p_diff(key1, key2))
    return list_diff


# Theoretic failure probability of noisy QSS protocol
def theoretic_fail_proba(p):
    # p : strength of QDC
    
    return p/2 * (p**2 - 3*p + 3)


# Plot failure probability of QSS protocol when noisy and denoised
def plot_noisy_VS_denoised_QSS(P, list_diff, list_diff_dn, theo_p, filename = None):
    # P : list of QDC strength
    # list_diff : list of failure probabilities of noisy QSS protocol
    # list_diff_dn : list of failure probabilities of denoised QSS protocol
    # theo_p : theoretic failure probability of noisy QSS protocol
    # filename : whether to save figure and figure name
    
    plt.plot(P, list_diff, '-o', color = 'r', ms = 4, label = 'Empirical probability')
    plt.plot(P, theo_p, color = '#0000CC', label = 'Theoretical probability')
    plt.plot(P, list_diff_dn, '-o', color = 'green', ms = 4, label = 'Denoised probability')
    plt.xlabel('Noise strength')
    plt.ylabel('Probability of having wrong shared bits')
    plt.grid(lw = 0.5, ls = 'dotted')

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.legend()
    
    if filename != None:
        plt.savefig(filename + '.pdf', bbox_inches = 'tight')
    


if __name__ == "__main__":
    
    # Create QAE and load weights
    QAE = QNN([3,1,3])
    K = np.load('/Users/tomachache/Downloads/Columbia/Spring 2020/Quantum Computing - Theory and Practice/Project/Saved_model/Depolarizing_channel/K-3,1,3. p = 0.2. 150 pairs, eps = 0.1. eta = 0.25. Depolarizing channel.npy')
    QAE.set_K(K)
    
    # Use LaTeX font for plots
    plt.rc('text', usetex = True)
    plt.rc('font', family = 'serif')

    # Plot noisy QSS circuit (with QAE)
    circ, _ = noisy_QSS_circuit(QAE = QAE)
    circ.draw(output = 'mpl', fold = 30).savefig('denoised_QSS.pdf', bbox_inches='tight')
    
    # Clean plot
    plt.clf()
    plt.close()
    
    # Get failure probability of protocol when noisy/denoised, for various noise strength of QDC
    P = np.linspace(0,1,11)
    nb_trials = 1000
    m = QAE.M[0]

    list_diff = noise_strength_VS_QSS(nb_trials, m = m, P = P)
    list_diff_dn = noise_strength_VS_QSS(nb_trials, m = m, QAE = QAE, P = P)
    theo_p = theoretic_fail_proba(P)
    
    # Get final figure
    plot_noisy_VS_denoised_QSS(P, list_diff, list_diff_dn, theo_p, filename = 'Noisy_QSS')