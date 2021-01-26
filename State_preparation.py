#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 18:40:07 2021

@author: tomachache
"""

from qiskit import *
import numpy as np

# Various state preparation
def state_preparation(m, name, p): 
    # m : nb of qubits 
    # name : name of the state we want 
    # p : proba associated with noise
    
    circ = QuantumCircuit(m, name = 'State prep')
    
    if name == 'GHZ':
        circ.h(0)
        for k in range(1,m):
            circ.cx(0,k)
    
    elif name == 'noisy_GHZ_bitflip':
        prob = np.random.rand(m)
        circ.h(0)
        for k in range(1,m):
            circ.cx(0,k)
            if prob[k] <= p: # flips each bit with proba p
                circ.x(k)
        if prob[0] <= p:
            circ.x(0)
            
    elif name == 'noisy_GHZ_QDC':
        probas = [1 - 3*p/4, p/4, p/4, p/4]
        gate_inds = np.random.choice(np.arange(4), size = m, p = probas)
        circ.h(0)
        for k in range(1,m):
            circ.cx(0,k)
            if gate_inds[k] == 1:
                circ.x(k)
            elif gate_inds[k] == 2:
                circ.y(k)
            elif gate_inds[k] == 3:
                circ.z(k)
        if gate_inds[0] == 1:
            circ.x(0)
        elif gate_inds[0] == 2:
            circ.y(0)
        elif gate_inds[0] == 3:
            circ.z(0)
            
    elif name == 'rigged_QDC': # QDC where 1st and 2nd qubits have different probas
        probas_rigged = [1-p, p/2, p/2, 0]
        probas_rigged2 = [1 - 29*p/30, 2*p/5, 2*p/5, p/6]
        probas = [1 - 3*p/4, p/4, p/4, p/4]
        gate_inds = np.random.choice(np.arange(4), size = m - 1, p = probas)
        gate_inds_r = np.random.choice(np.arange(4), p = probas_rigged)
        gate_inds_r2 = np.random.choice(np.arange(4), p = probas_rigged2)
        circ.h(0)
        circ.cx(0,1)
        if gate_inds_r2 == 1:
            circ.x(1)
        elif gate_inds_r2 == 2:
            circ.y(1)
        elif gate_inds_r2 == 3:
            circ.z(1)
        for k in range(2,m):
            circ.cx(0,k)
            if gate_inds[k-1] == 1:
                circ.x(k)
            elif gate_inds[k-1] == 2:
                circ.y(k)
            elif gate_inds[k-1] == 3:
                circ.z(k)
        if gate_inds_r == 1:
            circ.x(0)
        elif gate_inds_r == 2:
            circ.y(0)
        elif gate_inds_r == 3:
            circ.z(0)
    else:
        raise ValueError('Unrecognized name.')
            
    return circ