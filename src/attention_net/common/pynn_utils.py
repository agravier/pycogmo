#!/usr/bin/env python2
""" Functions and classes wrapping or complementing PyNN
functionality.
"""

from math import isnan
import numpy

import pyNN.nest as pynnn

class Weights(object):
    """Wraps a 2D array of floating-point numbers that has the same
    dimensions as the connectivity matrix between the two populations
    of neurons connected. Non-connected units i and j have
    weights[i][j] == NaN."""
    def __init__(self, weights_array):
        self._weights = numpy.array(weights_array)

    def __eq__(self, other):
        internalw = None
        if isinstance(other, numpy.ndarray):
            internalw = other
        elif isinstance(other, list):
            internalw = numpy.ndarray(other)
        elif not isinstance(other, Weights):
            return False
        else:
            internalw = other.numpy_weights
        if len(self.numpy_weights) != len(internalw):
            return False
        n_r = len(self.numpy_weights)
        for i in xrange(n_r):
            l = len(self.numpy_weights[i])
            if l != len(internalw[i]):
                return False
            for j in xrange(l):
                v1 = self.numpy_weights[i][j]
                v2 = internalw[i][j]
                if (isnan(v1) and isnan(v2)):
                    continue
                if v1 != v2:
                    return False
        return True

    @property
    def weights(self):
        return self._weights.tolist()

    @property
    def numpy_weights(self):
        return self._weights
    
    @weights.setter
    def weights(self, weights_array):
        if isinstance(weights_array, numpy.ndarray):
            self._weights = weights_array
        else:
            self._weights = numpy.array(weights_array)

    def adjust(self, error_mat, learning=0.01):
        """Returns a matrix adjusted by removing the error terms
        matrix from the array, scaling the change by the learning
        coerfficient (scalar) if available."""
        if isinstance(error_mat, Weights):
            error_mat = error_mat.numpy_weights
        return Weights(numpy.add(self._weights, -error_mat*learning))

def get_weights(proj):
    return Weights(proj.getWeights(format='array', gather=True))

def set_weights(proj, w):
    if isinstance(w, Weights):
        proj.setWeights(w.weights)

class InputLayer(object):
    """Wraps a 2D array of normalized floating-point numbers that
    has the same dimensions as the population of PyNN neurons to which
    it is to be presented. The stimulation scale can be adjusted by
    providing the max input amplitude in nA."""
    def __init__(self, input_array, max_namp):
        TODO
        source = pynnn.DCSource(amplitude=1.0, start=0.0, stop=None)
