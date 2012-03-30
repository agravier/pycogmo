#!/usr/bin/env python2
""" Functions and classes wrapping or complementing PyNN
functionality.
"""

import csv
import itertools
import functools
from math import isnan, ceil
import magic
import math
import numpy
from PIL import Image
import pyNN.nest as pynnn
import types

from utils import LOGGER, is_square


class InvalidFileFormatError(Exception):
    def __init__(self, mime_type, mime_subtype):
        self._type = mime_type
        self._subtype = mime_subtype

    def __str__(self):
        return "%s files of type %s are not supported." % \
            (self._type, self._subtype)


class InvalidMatrixShapeError(Exception):
    def __init__(self, req_dim1, req_dim2, prov_dim1, prov_dim2):
        self._req = req_dim1, req_dim2
        self._prov = prov_dim1, prov_dim2

    def __str__(self):
        return ("The required input data shape should be "
                "%s,%s, but the shape of the data provided is "
                "%s,%s.") % (self._req[0], self._req[1], \
                self._prov[0], self._prov[1])


class Weights(object):
    """Wraps a 2D array of floating-point numbers that has the same
    dimensions as the connectivity matrix between the two populations
    of neurons connected. Non-connected units i and j have
    weights[i][j] == NaN. l_rate can be given to set a different
    default learning rate than 0.01"""
    def __init__(self, weights_array, l_rate=0.01):
        self._weights = numpy.array(weights_array)
        self._default_l_rate = l_rate
        self._update_shape()

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

    def _update_shape(self):
        shape = self._weights.shape
        self._dim1 = shape[0]
        if len(shape) > 1:
            self._dim2 = shape[1]
        else:
            self._dim2 = 0

    @property
    def shape(self):
        return self._dim1, self._dim2

    @property
    def weights(self):
        return self._weights.tolist()

    @property
    def flat_weights(self):
        return list(itertools.chain.from_iterable(self._weights.tolist()))

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
        self._update_shape()

    def __getitem__(self, i):
        return self._weights[i]

    def __setitem__(self, key, value):
        self._weights[key] = value

    def adjusted(self, error_mat, learning=None):
        """Returns a matrix adjusted by removing the error terms
        matrix from the array, scaling the change by the learning
        coefficient (scalar) if available."""
        if learning == None:
            learning = self._default_l_rate
        if isinstance(error_mat, Weights):
            error_mat = error_mat.numpy_weights
        if not isinstance(error_mat, numpy.ndarray):
            error_mat = numpy.array(error_mat)
        return Weights(numpy.add(self._weights, -1 * error_mat * learning))


def get_weights(proj):
    return Weights(proj.getWeights(format='array'))


def set_weights(proj, w):
    if isinstance(w, Weights):
        proj.setWeights(w.flat_weights)
    else:
        raise TypeError("Requires an argument of class Weights.")


def read_input_data(file_path, dim1, dim2, m=None):
    """The libmagic file identifier can be passed as argument m (used for
    testing)."""
    if m == None:
        m = magic.Magic(mime=True)
    mime = m.from_file(file_path)
    mime = mime.lower().split('/')
    float_array = None
    if mime[0] == 'image':
        float_array = read_image_data(file_path)
    elif mime[0] == 'text':
        if mime[1] == 'plain':
            float_array = read_csv_data(file_path)
        else:
            raise InvalidFileFormatError(mime[0], mime[1])
    else:
        raise InvalidFileFormatError(mime[0], mime[1])
    verify_input_array(float_array, dim1, dim2)
    return float_array


def read_image_data(file_path):
    """Raises IOError if the file is not an image."""
    im = Image.open(file_path)
    # if im.size != (dim1, dim2):
    #     raise InvalidMatrixShapeError((dim1, dim2), im.size)
    byte_array = numpy.array(im.convert("L"))  # grayscale, [0 255]
    norm_array = byte_array / 255.
    return norm_array


def read_csv_data(file_path):
    """Raises IOError if the file is not a CSV file."""
    float_array = []
    try:
        with open(file_path, 'rb') as f:
            row_reader = csv.reader(f)
            for r in itertools.ifilter(None, row_reader):
                float_array.append(map(float, r))
        return numpy.array(float_array)
    except ValueError as e:
        raise IOError(str(e))


def verify_input_array(float_array, dim1, dim2):
    d1 = len(float_array)
    if d1 != dim1:
        raise InvalidMatrixShapeError(dim1, dim2, d1, "unkown")
    for r in float_array:
        d2 = len(r)
        if d2 != dim2:
            raise InvalidMatrixShapeError(dim1, dim2, d1, d2)
        real = numpy.isreal(r)
        if not isinstance(real, bool):
            real = real.all()
        if not real:  # row test
            raise TypeError("The input array contains invalid data.")


class InputSample(object):
    """Wraps a 2D array of normalized floating-point numbers that has
    the same dimensions as the InputLayer to which it is
    presented. The data can be an array, or copied from an object with
    [][] accessor, loaded from a file, uniformly initialized to the
    same value, or initialized by a user-provided function."""
    # implement an [][] accessor
    def __init__(self,  dim1, dim2, initializer, expand=True):
        """The initializer can be an array, an object with [][]
        accessor, a file path (string), a single floating point number
        within [0,1] (the array is uniformly initialized to the same
        value), or a user-provided callable that takes two integers x
        and y in [0, dim1[ and [0, dim2[ respectively, and returns the
        value to be stored in the array at [x][y]. The optional
        parameter expand affects the case where the initializer is a
        callable, an object with __getitem__, or a single number. In
        those case, setting expand to False prevents the
        precomputation of the whole array, and the InputSample
        accessor encapsulate the function call, the object accessor,
        or always returns the given number. If expand is True, the
        InputSample created is mutable. If expand is False, the
        InputSample is immutable."""
        self._array = []
        self._getitem = lambda k: self._array[k]
        self._setitem = self._assign_to_array
        if isinstance(initializer, basestring):
            try:
                self._array = read_input_data(initializer, dim1, dim2)
            except IOError as e:
                LOGGER.error("Could not read file %s.", initializer)
                raise e
        elif isinstance(initializer, types.FileType):
            raise TypeError("Pass a string with the filepath to the " 
                            "InputSample initializer, instead of a "
                            "file descriptor.")
        elif isinstance(initializer, list): 
            self._array = initializer
        elif hasattr(initializer, '__getitem__'):
            if expand:
                for x in xrange(dim1):
                    self._array.append([])
                    for y in xrange(dim2):
                        self._array[x].append(initializer[x][y])
            else:
                self._array = initializer
                self._setitem = self._raise_immutable
        elif hasattr(initializer, '__call__'): 
            # to restrict to functions:
            # isinstance(initializer, 
            #            (types.FunctionType, types.BuiltinFunctionType))
            if expand:
                for x in xrange(dim1):
                    self._array.append([])
                    for y in xrange(dim2):
                        self._array[x].append(initializer(x,y))
            else:
                class InitCont(object):
                    def __init__(self, x):
                        self._x = x
                    def __getitem__(self, y): 
                        return initializer(self._x, y)
                self._getitem = lambda x: InitCont(x)
                self._setitem = self._raise_immutable
        self._dim1 = dim1
        self._dim2 = dim2
        if expand:
            verify_input_array(self._array, dim1, dim2)

    def _raise_immutable(self, *args):
        raise TypeError("Attempted change of state on an "  
                        "immutable InputSample (created with "
                        "expand=False)")

    def _assign_to_array(self, k, v):
        self._array[k] = v

    def __getitem__(self, k):
        return self._getitem(k)

    def __setitem__(self, k, v):
        self._setitem(k, v)

    @property
    def shape(self):
        return self._dim1, self._dim2


class RectilinearLayerAdapter(object):
    """Base class adapting PyNN layers."""
    def __init__(self, pynn_pop, dim1, dim2):
        self.unit_adapters_mat = []
        for x in xrange(dim1):
            self.unit_adapters_mat.append([])
            for y in xrange(dim2):
                self.unit_adapters_mat[x].append([None, 
                                           pynn_pop[x*dim2+y]])
        self._dim1 = dim1
        self._dim2 = dim2
        self.pynn_population = pynn_pop

    @property
    def shape(self):
        return self._dim1, self._dim2

    def __getitem__(self, i):
        return self.unit_adapters_mat[i]


INPUT_LAYER_MAX_NAMP_DEFAULT = 100
class RectilinearInputLayer(RectilinearLayerAdapter):
    """Wraps a 2D array of electrodes with the same dimensions (dim1,
    dim2) as the PyNN population in which it injects current. The
    stimulation scale can be adjusted by providing the max input
    amplitude in nA."""
    def __init__(self, pynn_pop, dim1, dim2, max_namp=INPUT_LAYER_MAX_NAMP_DEFAULT):
        super(RectilinearInputLayer, self).__init__(pynn_pop, dim1, dim2)
        self.input_scaling = max_namp

    # DCSources have to be recreated each time.
    def apply_input(self, sample, start_time, duration,
                    max_namp = None, dcsource_class = pynnn.DCSource):
        """Given a sample of type InputSample and of same shape as the
        input layer, and a duration, creates and connects electrodes
        that apply the input specified by the input sample matrix to
        the input population. A max_namp value can be specified in
        nanoamperes to override the max current corresponding to an
        input value of 1 given at construction time. dcsource_class is
        here as a primitive dependency injection facility, for
        testing.""" 
        if max_namp == None:
            max_namp = self.input_scaling
        for x in xrange(self._dim1):
            for y in xrange(self._dim2):
                # Will the GC collect the electrodes? Does PyNN delete
                # them after use?
                self.unit_adapters_mat[x][y][0] = \
                    dcsource_class(amplitude=max_namp * sample[x][y], 
                                   start=start_time, 
                                   stop=start_time+duration)


class RectilinearOutputRateEncoder(RectilinearLayerAdapter):
    """Keeps track of the weighted averages on a sliding window of the
    output rates of all units in the topographically rectilinear
    population of units. The update period and can be overridden at update
    time."""
    # Default width of the sliding window in simulator time units. The
    # weight of past rates in activity calculation decreases linearly
    # so that it is 0 when window_width old, and 1 for sim.now()
    DEFAULT_WINDOW_WIDTH = 100;
    DEFAULT_UPDATE_PERIOD = 10
    def __init__(self, pynn_pop, dim1, dim2,
                 update_period = DEFAULT_UPDATE_PERIOD,
                 window_width=DEFAULT_WINDOW_WIDTH):
        super(RectilinearOutputRateEncoder, self).__init__(pynn_pop, dim1, dim2)
        self.window_width = window_width
        self.update_period = update_period
        # the number of records needs to be one more than requested
        # because we are interested in the rate of firing, which is
        # the difference in total number of spikes fired between now
        # and 1 update period ago. In general, we need n+1 data points
        # to determine n such differences.
        self.hist_len = int(ceil(self.window_width/self.update_period)) + 1
        for x in xrange(self._dim1):
            for y in xrange(self._dim2):
                self.unit_adapters_mat[x][y][0] = \
                    numpy.zeros(self.hist_len, dtype=numpy.int)
        self.idx = -1
        self.update_history = None # initialized at first update

    def extend_capacity(self, idx):
        """Adds one cell to all logging structures at position idx, and 
        increments self.hist_len."""
        for x in xrange(self._dim1):
            for y in xrange(self._dim2):
                self.unit_adapters_mat[x][y][0] = numpy.concatenate(
                    (self.unit_adapters_mat[x][y][0][:idx],
                     [-1],
                     self.unit_adapters_mat[x][y][0][idx:]))
        self.update_history = numpy.concatenate(
            (self.update_history[:idx], [-1], self.update_history[idx:]))
        self.hist_len += 1

    @staticmethod
    def make_hist_weights_vec(update_history, window_width, idx):
        """Parameters are the update times array, the rate averaging window 
        width, and the current time index in the update times array. 
        Returns the ndarray of weights by which to multiply the rates 
        history vector to calculate the weighted recent activity of the unit.
        The weight for the oldest rate is the head of the array. The sum of 
        weights is 1 if the update_history array covers at least the duration
        of window_width."""
        update_hist = numpy.append(update_history[idx+1:],
                                   update_history[:idx+1])
        update_dt = numpy.diff(update_hist)
        cumsum_dt = update_dt[::-1].cumsum()[::-1] # reversed cumulative sum
        last_t = update_hist[-1]
        cutoff_t = last_t - window_width
        l_h = 1 - cumsum_dt / (window_width * 1.)
        r_h = 1 - (numpy.append(cumsum_dt[1:], [0]) / (window_width * 1.))
        areas = numpy.fromiter(
            itertools.imap(lambda i, x:
                # in window -> area; out -> 0; border -> triangle
                (l_h[i] + r_h[i]) * update_dt[i] if x <= window_width 
                    else max(abs(r_h[i]) * (update_hist[i + 1] - cutoff_t), 0),
                itertools.count(), cumsum_dt),
            numpy.float)
        return areas / window_width
    
    def advance_idx(self):
        self.idx = self.next_idx

    @property
    def next_idx(self):
        return self.idx_offset(1)

    @property
    def previous_idx(self):
        return self.idx_offset(-1)

    def idx_offset(self, offset):
        """Returns the value of the index with the (positive or negative)
        offset added."""
        return (self.idx + offset) % self.hist_len
    
    # The data structure for the rate history of one unit is a
    # circular list of rates, and an integer index (self.idx, common
    # to all units) pointing to the most recent record. The size of
    # this list is determined in __init__ by the window_width and
    # update_period. Each unit's history is kept in the
    # RectilinearLayerAdapter's unit_adapters_mat[x][y][0]. There is
    # an additional circular list of updates timestamps for testing.

    # We assume that the necessary recorders have been set up.
    def update_rates(self, t_now):
        """t_now is the timestamp for the current rates being recorded."""
        if self.idx != -1:
            # Not the first update, so the state is consistent.
            if t_now - self.update_history[self.idx] < self.update_period:
                # Premature update -> we may need to increase the arrays length
                # to have enough place to cover the full window width.
                # The total time covered by the rate log after idx increment
                # will be: 
                total_covered_dt = t_now - \
                    self.update_history[self.next_idx]
                if total_covered_dt < self.window_width:
                    # The arrays are insufficient to cover the whole window
                    # width. We need to extend all arrays by one (add one entry
                    # to all logging structures).
                    self.extend_capacity(self.next_idx)
        else:
            # First update:
            # Initialize the update times log to past values to have a 
            # consistent state without having to wait for the whole update
            # window to have been crawled once. 
            self.update_history = t_now - self.update_period * \
                numpy.array([0] + range(self.hist_len-1, 0, -1))
        self.advance_idx()
        self.update_history[self.idx] = t_now
        rec = self.pynn_population.get_spike_counts();
        for x in xrange(self._dim1):
            for y in xrange(self._dim2):
                self.unit_adapters_mat[x][y][0][self.idx] = \
                    rec.get(self.pynn_population[x*self._dim2+y])

    def get_rates(self):
        r = numpy.zeros((self._dim1, self._dim2), dtype=numpy.float)
        for x in xrange(self._dim1):
            for y in xrange(self._dim2):
                r[x][y] = self.f_rate(self.unit_adapters_mat[x][y][0])
        return r

    def f_rate(self, np_a):
        """Returns the weighted average of the rates recorded in the
        differences of the array np_a."""
        rates = numpy.diff(numpy.append(np_a[self.idx+1:], np_a[:self.idx+1]))
        return self.make_hist_weights_vec(self.update_history, 
                                          self.window_width, 
                                          self.idx).dot(rates)


# WARNING / TODO: The following function reveals a design flaw. PyNN is insufficient and its networks should be encapsulated along with more metadata.
def population_adpater_provider(pop_prov_dict,
                                provided_class,
                                population):
    """Factory function providing an adapter of the specified class for
    the population parameter. pop_prov_dict is a dictionary taking a
    (population, provided_class) tuple as key, and returning an
    instance of provided_class initialized with 3 arguments: the population,
    its size in the first dimension, and its size in the second dimension."""
    key = (population, provided_class)
    if pop_prov_dict.has_key(key):
        return pop_prov_dict[key]
    else:
        LOGGER.warning(("No %s for population %s, creating one assuming" 
                       "a square shape."), provided_class.__name__,
                       population.label)
        if not is_square(population.size):
            raise TypeError("The input layer shape could not be guessed.")
        dim = int(math.sqrt(population.size))
        inst = provided_class(population, dim, dim)
    return pop_prov_dict.setdefault(key, inst)


POP_ADAPT_DICT = {}


get_input_layer = functools.partial(population_adpater_provider,
                                    POP_ADAPT_DICT,
                                    RectilinearInputLayer)
get_input_layer.__doc__ = ("Provides a unique input layer for the"
                           "given population.") 


get_rate_encoder = functools.partial(population_adpater_provider,
                                     POP_ADAPT_DICT,
                                     RectilinearOutputRateEncoder) 
get_rate_encoder.__doc__ = ("Provides a unique rectilinear output rate "
                            "encoder for the given population.")


def enable_recording(*p):
    """Turns on spike recorders for all populations in parameter"""
    for pop in p:
        pop.record(to_file=False)

