# !pip install pygsp==0.5.1

from pygsp import graphs, filters, plotting
import collections
import numpy.matlib
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os

root = os.getcwd()


def flatten(weights):
    """
    Flatten the parameters into a vector.

    Parameters:
        weights (list of numpy arrays): The model weights.

    Returns:
        np.ndarray: A 1D numpy array containing all the flattened parameters.
    """
    return np.concatenate([w.flatten() for w in weights])

def unflatten(flat_w, old_w):
  """
Use to unflatten the weights of the network
  """
# """ flat_w : 1D array of weights (flattened).old_w : output of model.get_weights(). """
  new_w = [ ]
  
  i = 0
  for layer in old_w:
    size = layer.size
    new_w.append(flat_w[i:i+size].reshape(layer.shape))
    i += size

  return new_w 
