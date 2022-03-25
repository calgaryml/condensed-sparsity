import torch
import numpy as np
import os
import sys



def spe_matmul(inputs, weights, indx_seqs):

    """
    Performs the cond-matmul operation. 

    Args:
        inputs:  2d array, batch of input vectors, shape (batch_size, input_len)
        weights: 2d array, weight matrix, shape (num_units, fan_in)
        indx_seqs: 2d array, sequences of input vector indices, shape (num_units, fan_in)

    Returns:
        Layer output, shape (batch_size, num_units)
    """

    assert indx_seqs.shape==weights.shape

    # for each set of recombination vectors:
    # element-wise multiplication along the rows and summation of all vectors
    # --> output has shape (batch_size, num_units)
    v_out= torch.sum(weights * inputs[:, indx_seqs], axis=2)

    return v_out



def gen_indx_seqs(num_in, num_out, input_len, fan_out_const):
    """
    Generates indices for drawing recombination vectors from the input vector v.
        
        number recomb.vectors = num_in (= fan_in, also called k)
        length of each recomb.vector = num_out (= n2)

    Args:
        weights: 2d array, condensed weight matrix, shape (num_out, num_in)
        input_len: int, length of the input vector
        fan_out_const: bool, if True, nearly constant fan-out will be ensured
        
    Returns:
        A 2d array of indices of the same shape as weights.
    """
    
    #num_out, num_in= weights.shape

    # init index sequences
    #indx_seqs= np.zeros((weights.shape))
    indx_seqs= np.zeros(( num_out, num_in ))

    # indices of v (the input of length d)
    v_inds= np.arange(input_len)

    # initialize an array of probabs for every index of v (initially uniform)
    probs= 1/input_len*np.ones(input_len)

    for row_nr in range(num_out):
        chosen_inds= np.random.choice( v_inds, size=num_in, replace=False, 
                                       p= probs/sum(probs) )
        chosen_inds.sort()
        # update probabs only if want to control fan_out
        if fan_out_const: 
            probs[chosen_inds]/= (100*input_len)

        indx_seqs[row_nr,:]= chosen_inds
    
    return indx_seqs.astype(int)


