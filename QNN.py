#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 18:21:20 2021

@author: tomachache
"""

from qiskit import *
import numpy as np
import time
from scipy import linalg
from qiskit.extensions import UnitaryGate


# Create Pauli matrices (base for unitary matrices used in QNNs)
def create_Pauli():
    I = np.eye(2)
    X = np.array([[0,1], [1,0]])
    Y = 1j * np.array([[0,-1], [1,0]])
    Z = np.array([[1,0], [0,-1]])

    return [I, X, Y, Z]
    

# Create all possible tensor products of m Pauli matrices
def tensor_Paulis(m):
    Pauli = create_Pauli()
    
    def get_tensors(m, U):
        if m == 0:
            return np.array(U)
        else:
            return get_tensors(m-1, [np.kron(u, P) for u in U for P in Pauli])
        
    return get_tensors(m, [np.eye(1)])


# Create a QNN
class QNN:
    def __init__(self, M): # M : topology of the network ; e.g. M = [4,2,1,2,4]
        self.M = M
        self.num_layers = len(M)
        
        # Compute the width of the network (nb of qubits used in subroutine 2)
        w = 0
        for i in range(self.num_layers - 1):
            w = max(w, self.M[i] + self.M[i+1])
        self.W = w # width
        
        self.num_qubits = 1 + self.M[0] + self.W # total nb of qubits required
        
        # Creating K (vector of all coefficients to learn)
        
        # First compute the nb of coeffs needed
        self.nb_coeffs = 0
        
        # Create and store the "basis" of gates' matrices (all the different tensor products of length M[i-1]+1 of Pauli's)
        # These basis are then multiplied with coeffs K, summed, and transform in a unitary U = e^{iS} then into a gate
        self.mat_basis = []
        
        for i in range(1, self.num_layers):
            self.nb_coeffs += self.M[i] * 4**(self.M[i-1]+1)
            self.mat_basis.append(tensor_Paulis(self.M[i-1]+1))
        
        # Choose required initialization for K
        self.K = np.zeros(self.nb_coeffs)           # initialize at 0
        #self.K = np.random.rand(self.nb_coeffs)    # initialize with random values in [0,1]
        
        # Choose backend and options
        self.backend = Aer.get_backend('qasm_simulator')
        self.backend_options = {'max_parallel_experiments': 0, 'max_parallel_threads': 0}
        
        # Choose nb of shots (S)
        self.shots = 1000

        self.circ = None # record current circuit used
        self.show_circ = None # record compiled version of circuit (for plotting purpose)
        
        self.create_circuit() # create circuit
        
    def set_K(self, K): # set pre-learned K
        assert len(K) == self.nb_coeffs
        self.K = K
        self.create_circuit()
        return
        
    def subroutine_2(self):
        circ = QuantumCircuit(self.W, name = 'Quantum AE')
        
        free_qubits = self.M[0] # will allow to see which qubits are "free" for the next layer
        m_new = np.array(range(free_qubits)) # m_new is the list of "free" qubits
    
        cpt = 0
        for i in range(1, self.num_layers):
            m_old = list(m_new) # m_old is the list of qubits "taken" (i.e. used by the current layer)
            m_new = np.array(range(free_qubits, free_qubits + self.M[i])) % self.W
            free_qubits += self.M[i]
            for j in range(self.M[i]):
                # Create unitary gates by multipliying K with the basis of gates' matrices
                # self.mat_basis[i-1] : all the basis matrices for layer i
                # self.K[j*(4**(self.M[i-1]+1)) + cpt : (j+1)*(4**(self.M[i-1]+1)) + cpt] : relevant coeffs
                # multiply these 2, sum to get a matrix S, perform e^{iS}, then create the gate
            
                C_U = UnitaryGate(linalg.expm(1j * np.sum(self.mat_basis[i-1] * self.K[j*(4**(self.M[i-1]+1)) + cpt : (j+1)*(4**(self.M[i-1]+1)) + cpt, None, None], axis = 0)))
                circ.append(C_U, m_old + [m_new[j]])
            
            cpt += self.M[i] * 4**(self.M[i-1]+1)
            
            circ.reset(m_old)
            if i < self.num_layers - 1: # don't put barrier at the end since there will be one already
                circ.barrier() # if we want to set a barrier between each layer (not really useful) 
            # if we put barriers be careful as the positions of gates in the fit function will change
        
        return circ, m_new # that way we have the final relevant qubits
        
    def create_circuit(self): # create the circuit corresponding to the network
        circ = QuantumCircuit(self.num_qubits, 1)
        
        # Initialize empty states preparation
        input_state = QuantumCircuit(self.M[0], name = 'Input State Prep')
        target_state = QuantumCircuit(self.M[0], name = 'Target State Prep')
        circ.append(input_state.to_instruction(), range(1 + self.M[0], 1 + 2*self.M[0]))
        circ.append(target_state.to_instruction(), range(1, 1 + self.M[0]))
        circ.barrier()
        
        # Subroutine 2
        sub, out_qubits = self.subroutine_2() 
        circ.append(sub.to_instruction(), range(1 + self.M[0], self.num_qubits))
        circ.barrier()
        
        # Subroutine 1
        circ.h(0)
        for k in range(self.M[0]):
            circ.cswap(0, k + 1, 1 + self.M[0] + out_qubits[k])
        circ.h(0)

        circ.barrier()
        circ.measure(0,0)
        self.show_circ = circ
        self.circ = circ.decompose()
        
    def run(self): # run the circuit and output the fidelity
        result = execute(self.circ, self.backend, shots = self.shots).result()
        #result = execute(self.circ, self.backend, shots = self.shots, backend_options = self.backend_options).result()
        return (2 * result.get_counts(0)['0']/self.shots - 1)
    
    def run_multiple_circs(self, circs, batch_size, train = True): # run multiple circuits in batches
        n = len(circs)
        result = execute(circs, self.backend, shots = self.shots).result()
        #result = execute(circs, self.backend, shots = self.shots, backend_options = self.backend_options).result()

        if train:
            assert n == batch_size * (self.nb_coeffs + 1)
            return np.mean([[2 * result.get_counts(i * batch_size + j)['0']/self.shots - 1 for j in range(batch_size)] for i in range(self.nb_coeffs + 1)], axis = 1) # nb_coeff + 1 to account for the original cost
        else: # we're calling this function in the test phase
            assert n == batch_size
            return np.array([2 * result.get_counts(i)['0']/self.shots - 1 for i in range(n)])

    def fit(self, training_states, nb_epochs, epsilon, eta, batch_size_train = None, batch_size_test = None,
            goal_state = None, validation_states = None, use_Adam = False, use_momentum = False, 
            run_in_batch = False, save_best_val_fid = True): # train the network
        # training_states : list of pairs of circuit preparing input/target states
        # nb_epochs : nb of rounds 
        # epsilon : step used for the derivative approximation
        # eta : learning rate
        # batch_size_train/test : nb of states considered at each epoch for the training/testing at the end
        # goal_state : the 'goal' state we want, when we use the network as an AE (just to see if the network is improving at every step)
        # For instance, for the AE considered, the training_states will be pairs of noisy GHZ while the goal state is the true GHZ 
        # validation_states : states similar to input states (come from same noisy distribution), used for validation
        # use_Adam/momentum : whether to use Adam gradient ascent or SGD with momentum
        # run_in_batch : whether to run all the circuits in batches (may speed up computation, but not realistic physically)
        # save_best_val_fid : whether to save weights corresponding to maximum fidelity on validation states
        
        if validation_states is not None:
            assert goal_state is not None # validation states are used with the goal state
            
        if use_Adam: # Implement Adam gradient descent (ascent here)
            beta_1 = 0.9
            beta_2 = 0.999
            eps = 10**(-8)
            t = 0
            m = np.zeros(self.nb_coeffs)
            v = np.zeros(self.nb_coeffs)
        if use_momentum: # Use SGD with momentum
            velocity = np.zeros(self.nb_coeffs)
            mu = 0.9
        
        N = len(training_states) # nb of training pairs
        
        if batch_size_train is None:
            batch_size_train = N
        
        C = [] # will save the cost (i.e. fidelity) after every epoch
        C_val = [] # same for validation states
        
        for epoch in range(nb_epochs):
            start = time.time()
            
            batch = np.random.choice(N, batch_size_train, replace = False) # pick the batch
            
            circs = [] # will be used in case run_in_batch = True
            
            cost = 0
            for k in batch: # replace the states preparations by the current one
                # target state is the first gate and input is the second one
                self.circ.data[1] = (training_states[k][0], self.circ.data[1][1], self.circ.data[1][2])
                self.circ.data[0] = (training_states[k][1], self.circ.data[0][1], self.circ.data[0][2])
                if run_in_batch:
                    circs.append(self.circ.copy())
                else:
                    cost += self.run()
            cost /= batch_size_train
        
            delta = np.zeros(self.nb_coeffs)
            
            cpt = 0
            pos_gate = 0 # record position of the gate we are currently modifying
            for i in range(1, self.num_layers):
                for j in range(self.M[i]): # nb of unitary gate at that layer
                    for v in range(4**(self.M[i-1]+1)): # nb of coeff of each gate of the current layer
                        # K is the vector of all coefficients (i.e. coeffs of all unitary matrices)
                        # K[v + j*(4**(self.M[i-1]+1)) + sum_{s = 1}^{i-1} (self.M[i] * 4**(self.M[i-1]+1))] is the v-th coeff of the j-th Unitary of the i-th layer
                        # Total nb of operations for backprop : nb_coeffs = sum_{i=1}^L M[i] * 4**(M[i-1]+1)
                        self.K[v + j*(4**(self.M[i-1]+1)) + cpt] += epsilon
                        
                        # Instead of updating whole circuit : pop 1 gate then recreate it
                        # j-th gate (from 0 to M[i]-1) of the i-th (from 1 to L-1) layer is at position : j + sum_{s = 1}^{i-1} M[s]
                        # (without resets nor barriers)
                        # We have M[i-1] resets at the end of layer i (from 1 to L-1)
                        # So (without barriers) j-th gate of the i-th layer is at position : j + sum_{s = 1}^{i-1} (M[s]+M[s-1])
                        # Create a counter to record the gate's position : pos_gate = sum_{s = 1}^{i-1} (M[s]+M[s-1])
                        # Add + 2 to the position to account for state preparations
                        # + i for barriers
                        
                        self.circ.data[j + pos_gate + 2 + i][0].params[0] = linalg.expm(1j * np.sum(self.mat_basis[i-1] * self.K[j*(4**(self.M[i-1]+1)) + cpt : (j+1)*(4**(self.M[i-1]+1)) + cpt, None, None], axis = 0))
                        
                        self.K[v + j*(4**(self.M[i-1]+1)) + cpt] -= epsilon
                        
                        # Compute the new avg cost
                        new_cost = 0
                        for k in batch: # insert the state preparations, run the network, and delete them
                            self.circ.data[1] = (training_states[k][0], self.circ.data[1][1], self.circ.data[1][2])
                            self.circ.data[0] = (training_states[k][1], self.circ.data[0][1], self.circ.data[0][2])
                            
                            if run_in_batch:
                                circs.append(self.circ.copy())
                            else:
                                new_cost += self.run()
                        new_cost /= batch_size_train
                        
                        delta[v + j*(4**(self.M[i-1]+1)) + cpt] = (new_cost - cost)/epsilon
                cpt += self.M[i] * 4**(self.M[i-1]+1)
                pos_gate += self.M[i] + self.M[i-1]
            
            if run_in_batch:
                delta_0 = self.run_multiple_circs(circs, batch_size_train)
                delta = (delta_0[1:] - delta_0[0])/epsilon
            
            if use_Adam:
                # Adam update rule
                m = beta_1 * m + (1 - beta_1) * delta
                v = beta_2 * v + (1 - beta_2) * (delta**2)
                t += 1
                self.K += eta * (m/(1 - beta_1**t)) / (np.sqrt(v/(1 - beta_2**t)) + eps)
            elif use_momentum:
                # add momentum
                velocity = mu * velocity + eta * delta
                self.K += velocity
            else:
                # regular update rule
                self.K += eta * delta
            
            # Now we re-create the circuit --> maybe faster to just re-run subroutine 2 and only replace this part
            self.create_circuit()
            
            if batch_size_test is None:
                batch_size_test = N
            
            # Do a final run to compute the new cost (for validation purpose)
            batch_test = np.random.choice(N, batch_size_test, replace = False)
            final_cost = 0
            test_circs = []
            
            if validation_states is not None:
                N_val = len(validation_states)
            
            val_cost = 0
            val_circs = []
            for k in batch_test: # insert the state preparations, run the network, and delete them
                self.circ.data[1] = (training_states[k][0], self.circ.data[1][1], self.circ.data[1][2])  
                if goal_state is not None : 
                    self.circ.data[0] = (goal_state, self.circ.data[0][1], self.circ.data[0][2])
                else:
                    self.circ.data[0] = (training_states[k][1], self.circ.data[0][1], self.circ.data[0][2])              
                if run_in_batch:
                    test_circs.append(self.circ.copy())
                else:
                    final_cost += self.run()
                    
            if validation_states is not None:
                for k in range(N_val):
                    self.circ.data[1] = (validation_states[k], self.circ.data[1][1], self.circ.data[1][2])  
                    self.circ.data[0] = (goal_state, self.circ.data[0][1], self.circ.data[0][2])             
                    if run_in_batch:
                        val_circs.append(self.circ.copy())
                    else:
                        val_cost += self.run()
                    
            if run_in_batch:
                final_cost = np.mean(self.run_multiple_circs(test_circs, batch_size_test, train = False))
            else:
                final_cost /= batch_size_test
            C.append(final_cost)
                
            if validation_states is not None:
                if run_in_batch:
                    val_cost = np.mean(self.run_multiple_circs(val_circs, N_val, train = False))
                else:
                    val_cost /= N_val  
                C_val.append(val_cost)
            
                if save_best_val_fid and val_cost == np.max(C_val): # save weights corresponding to best epoch on val set
                    np.save('K.npy', self.K)
                
            end = time.time()
            print('Epoch {} | Fidelity : {:.3f} | Validation Fidelity : {:.3f} | Time : {} s.'.
                  format(epoch + 1, final_cost, val_cost, int(end-start)))
            
            #if (epoch + 1) % 10 == 0: # allows to stop simulations without loosing everything if they are too long
             #   np.save('train_fid.npy', np.array(C))
              #  np.save('val_fid.npy', np.array(C_val))
            
        return C, C_val
    
    def test(self, test_states, run_in_batch = False): # test the network
        # test_states : list of pairs of circuit preparing input/target states
        N = len(test_states)
        
        cost = []
        circs = []
        for k in range(N): # insert the state preparations, run the network, and delete them
            self.circ.data[1] = (test_states[k][0], self.circ.data[1][1], self.circ.data[1][2])
            self.circ.data[0] = (test_states[k][1], self.circ.data[0][1], self.circ.data[0][2])  
            if run_in_batch:
                circs.append(self.circ.copy())
            else:
                cost += [self.run()]
        if run_in_batch:
            cost = self.run_multiple_circs(circs, N, train = False)
        else:
            cost = np.array(cost)
        return cost