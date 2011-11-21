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
    weights[i][j] == NaN. l_rate can be given to set a different
    default learning rate than 0.01"""
    def __init__(self, weights_array, l_rate = 0.01):
        self._weights = numpy.array(weights_array)
        self._default_l_rate = l_rate

    def __eq__(self, other):
        internalw = None
        if isinstance(other, numpy.ndarray):
            internalw = other
        elif isinstance(other, list):
            internalw = numpy.array(other)
        elif not isinstance(other, Weights):
            return False
        else:
            internalw = other.numpy_weights
        if len(numpy.atleast_1d(self.numpy_weights)) != \
                len(numpy.atleast_1d(internalw)):
            return False
        n_r = len(self.numpy_weights)
        for i in xrange(n_r):
            l = len(numpy.atleast_1d(self.numpy_weights[i]))
            if l != len(numpy.atleast_1d(internalw[i])):
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
        elif isinstance(weights_array, list):
            self._weights = numpy.array(weights_array)
        elif isinstance(weights_array, Weights):
            self._weights = weights_array.numpy_weights
        else:
            raise TypeError("Weights can be assigned to " 
                            "numpy.ndarray, common.pynn_utils.Weights,"
                            " or list types.") 

    def __getitem__(self, i):
        return self._weights[i]

    def __setitem__(self, key, value):
        self._weights[key] = value

    def adjusted(self, error_mat, learning=None):
        """Returns a matrix adjusted by removing the error terms
        matrix from the array, scaling the change by the learning
        coerfficient (scalar) if available."""
        if learning == None:
            learning = self._default_l_rate
        if isinstance(error_mat, Weights):
            error_mat = error_mat.numpy_weights
        if not isinstance(error_mat, numpy.ndarray):
            error_mat = numpy.array(error_mat)
        return Weights(numpy.add(self._weights, -1*error_mat*learning))

def get_weights(proj):
    return Weights(proj.getWeights(format='array', gather=True))

def set_weights(proj, w):
    if isinstance(w, Weights):
        proj.setWeights(w.weights)

class RectilinearInputLayer(object):
    """Wraps a 2D array of electrodes with the same dimensions (dim1,
    dim2) as the PyNN population in which it injects current. The
    stimulation scale can be adjusted by providing the max input
    amplitude in nA."""
    def __init__(self, target_pynn_pop, dim1, dim2, max_namp):
        self.electrodes = []
        for x in xrange(dim1):
            self.electrodes.append([])
            for y in xrange(dim2):
                self.electrodes[x].append([None, 
                                           target_pynn_pop[x*dim2+y]])
        self.input_scaling = max_namp
        self._dim1 = dim1
        self._dim2 = dim2
        self.pynn_population = target_pynn_pop
    
    @property
    def shape(self):
        return self._dim1, self._dim2

    def __getitem__(self, i):
        return self.electrodes[i]

    # DCSources have to be recreated each time.
    def apply_input(self, sample, start_time, duration, max_namp = None):
        """Given a sample of type InputSample and of same shape as the
        input layer, and a duration, creates and connects electrodes
        that apply the input specified by the input sample matrix to
        the input population. A max_namp value can be specfied in
        nanoamperes to override the max current corresponding to an
        input value of 1 given at construction time."""
        if max_namp == None:
            max_namp = self.input_scaling
        for x in xrange(self._dim1):
            for y in xrange(self._dim2):
                # Will the GC collect the electrodes? Does PyNN delete
                # them after use?
                self.electrodes[x][y][0] = \
                    pynnn.DCSource(amplitude=max_namp * sample[x][y], 
                                   start=start_time, 
                                   stop=start_time+duration)

class InputSample(object):
    """Wraps a 2D array of normalized floating-point numbers that has
    the same dimensions as the InputLayer to which it is presented."""
    # implement an [][] accessor
    pass

