#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 18:34:59 2021

@author: tomachache
"""

import numpy as np
import matplotlib.pyplot as plt
from QNN import QNN
from State_preparation import state_preparation


# Plot training fidelity
def plot_fid(train_fid, val_fid = None, filename = None):
    # train_fid : list of training fidelities at each epoch
    # val_fid : list of validation fidelities at each epoch
    # filename : whether to save figure and figure name
    
    nb_epochs = len(train_fid)
    E = [k+1 for k in range(nb_epochs)]
    
    plt.plot(E, train_fid, '-o', color = '#0000CC', ms = 4, label = 'Training pairs')
    if val_fid is not None:
        plt.plot(E, val_fid, '-o', color = 'red', ms = 4, label = 'Validation pairs')
        plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Training Fidelity')
    plt.xlim(1, nb_epochs)
    plt.ylim(ymax=1)
    plt.grid(lw = 0.5, ls = 'dotted')
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    if filename != None:
        plt.savefig(filename + '.pdf', bbox_inches = 'tight')


if __name__ == "__main__":
    
    # Create QNN
    my_QNN = QNN([2,1,2])
    
    # Set parameters
    proba_bitflip = 0.2
    nb_train_states = 50
    nb_val_states = 50
    nb_test_states = 50
    
    # Get training and validation states
    training_states = [(state_preparation(my_QNN.M[0], 'noisy_GHZ_bitflip', proba_bitflip), state_preparation(my_QNN.M[0], 'noisy_GHZ_bitflip', proba_bitflip)) for _ in range(nb_train_states)]
    val_states = [state_preparation(my_QNN.M[0], 'noisy_GHZ_bitflip', proba_bitflip) for _ in range(nb_val_states)]
    
    # Test the fidelity before training
    test_states = [(state_preparation(my_QNN.M[0], 'noisy_GHZ_bitflip', proba_bitflip), state_preparation(my_QNN.M[0], 'GHZ', 0)) for _ in range(nb_test_states)]
    init_fid = my_QNN.test(test_states)
    print('Fidelity before training : {:.2f}'.format(np.mean(init_fid)))
    
    # Train the QNN
    nb_epochs = 3
    epsilon = 0.1
    eta = 1/4
    
    train_fid, val_fid = my_QNN.fit(training_states, nb_epochs, epsilon, eta, 
                           goal_state = state_preparation(my_QNN.M[0], 'GHZ', 0), 
                           validation_states = val_states)#, run_in_batch = True)
    
    # Save training and validation fidelities
    #np.save('train_fid.npy', np.array(train_fid))
    #np.save('val_fid.npy', np.array(val_fid))
    
    # Test the fidelity after training
    final_fid = my_QNN.test(test_states)
    print('Fidelity after training : {:.2f}'.format(np.mean(final_fid)))
    
    # Use LaTeX font for plots
    plt.rc('text', usetex = True)
    plt.rc('font', family = 'serif')
    
    # Get plot of training/validation fidelity
    plot_fid(train_fid, val_fid, filename = 'fid_VS_epochs')