#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 18:03:41 2021

@author: tomachache
"""

from qiskit import *
import numpy as np


# Create regular QSS circuit
def QSS_circuit(m = 3):
    # m : nb of qubit
    
    circ = QuantumCircuit(m, m, name = 'QSS')
    
    # Initialize the GHZ triplet
    circ.h(0)
    for k in range(1,m):
        circ.cx(0,k)
    circ.barrier()
    
    # Each participant make a measurement according to the X or Y basis at random
    p = np.random.rand(m)
    basis = '' # will record the basis we chose for measurement
    for k in range(m):
        if p[k] < 1/2: # make a measurement in the X basis
            basis += 'X'
            circ.h(k)
        else: # make a measurement in the Y basis
            basis += 'Y'
            circ.sdg(k) # S^\dagger
            circ.h(k)
    circ.barrier()
    
    circ.measure(range(m), range(m))
    return circ, basis


# Simulate regular QSS protocol
def QSS_protocol(nb_trials, m = 3):
    # nb_trials : nb of times we simulate protocol
    # m : nb of qubits
    
    Valid_basis = ['XXX', 'YYX', 'XYY', 'YXY']
    
    key = '' # final key
    
    for _ in range(nb_trials):
        circ, basis = QSS_circuit(m) # create new QSS circuit
    
        if basis not in Valid_basis:
            print('Invalid basis. Too bad !')
        else: # Alice, Bob and Charlie will be able to share a secret bit
            result = execute(circ, Aer.get_backend('qasm_simulator'), shots = 1).result()
            res = next(iter(result.get_counts())) # get first (and only) key --> resulting bits
            Alice = int(res[0])
            Bob = int(res[1])
            Charlie = int(res[2])
            
            if basis == Valid_basis[0]: # XXX 
                assert (Alice + Bob) % 2 == Charlie
                key += str(Charlie)
            else: # YYX, XYY or YXY
                assert (Alice + Bob + 1) % 2 == Charlie
                key += str(Charlie)
    print('Found secret key of length {}.'.format(len(key)))
    return key


if __name__ == "__main__":
    
    key = QSS_protocol(50)
    print(key)