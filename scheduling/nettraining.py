#!/usr/bin/env python2

# Copyright 2011, 2012 Alexandre Gravier (al.gravier@gmail)

# This file is part of PyCogMo.
# PyCogMo is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# PyCogMo is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with PyCogMo.  If not, see <http://www.gnu.org/licenses/>.

""" Implements algorithms that schedule the training of the PyNN
network.
"""

from SimPy.Simulation import Process

from common.pynn_utils import get_weights, set_weights, get_input_layer, \
    get_rate_encoder
from common.utils import log_tick, splice, LOGGER
from scheduling.pynn_scheduling import run_simulation, \
    schedule_input_presentation, schedule_output_rate_calculation
import itertools
import heapq
import numpy

# epoch: 1 series of presentations of the whole set
# presentation: presentation of one sample during a certain number of
# cycles or a certain simulated duration
# cycle: 1 atomic timestep in the network simulator


def _make_so_learning_rule(description, lambda_expr):
    """Returns a function that implements the learning algorithm given in
    lambda_expr. lambda_expr should take the parameters: pre-synaptic
    output(s), post-synaptic activation, weight(s), learning rate."""
    def f(pre_syn_out, post_syn_act, weights, l_rate):
        if isinstance(weights, (list, tuple)):
            weights = numpy.array(weights)
        if isinstance(pre_syn_out, (list, tuple)):
            pre_syn_out = numpy.array(pre_syn_out)
        return lambda_expr(pre_syn_out, post_syn_act, weights, l_rate)
    f.__doc__ = (description + " Parameters weights and pre_syn_out can be"
    " scalars, lists, vectors, or equivalent numpy structures. Returns a"
    " scalar or a numpy array.")
    return f

hebb_learning = _make_so_learning_rule("Hebbian learning.",
    lambda pre_syn_out, post_syn_act, weights, l_rate :
        weights + l_rate * post_syn_act * pre_syn_out)

oja_learning = _make_so_learning_rule(
    "Oja 1982, http://www.scholarpedia.org/article/Oja_learning_rule.",
    lambda pre_syn_out, post_syn_act, weights, l_rate :
        weights + l_rate * (post_syn_act * pre_syn_out - \
                            (post_syn_act ** 2) * weights))

conditional_pca_learning = _make_so_learning_rule("O'Reilly 2000, CECN.",
    lambda pre_syn_out, post_syn_act, weights, l_rate :
        weights + l_rate * post_syn_act * (pre_syn_out - weights))


def train_kwta(trained_population,
               input_population,
               input_samples,
               num_winners,
               neighbourhood_fn,
               presentation_duration,
               learning_rule,
               learning_rate,
               stop_condition):
    """Self-organized learning. The trained_population should have
    k-WTA compatible lateral inhibition. The neighbourhood function is
    used for neighbourhood learning only, not for inhibition. The
    neighbourhood function takes a population and a unit and returns a
    list of (unit, weight). As a guideline, num_winners should be at
    least equal to the number of elementary features (elementary as in
    encodable by one unit) present in each training sample."""
    while not stop_condition:
        kwta_epoch(trained_population, input_population, input_samples, num_winners,
                   neighbourhood_fn, presentation_duration, learning_rule,
                   learning_rate)
    

def kwta_epoch(trained_population,
               input_population,
               input_samples,
               num_winners,
               neighbourhood_fn,
               presentation_duration,
               learning_rule,
               learning_rate):
    rate_enc = get_rate_encoder(trained_population)
    for s in samples:
        kwta_presentation(trained_population, s, duration, k)
        argwinners = select_kwta_winners(trained_population, k)
        for w in argwinners:
            unit = rate_enc[w[1]][w[0]]
            # Adapt the weights to w 
            for n in neighbourhood_fn(trained_population, unit):
                pass # TODO  


def kwta_presentation(trained_population, input_population, sample, duration):
    schedule_input_presentation(input_population, sample, None, duration)
    schedule_output_rate_calculation(trained_population, None, duration)
    run_simulation()


def _infinite_xrange():
    i = 0
    while True:
        yield i
        i += 1


def select_kwta_winners(population, k):
    """Returns the list of coordinates of the k most active units in
    the population. Ties are broken using uniform random selection."""
    argwinners = []
    if k > 0:
        rate_enc = get_rate_encoder(population)
        rates = list(itertools.izip(splice(rate_enc.get_rates()),
                                     _infinite_xrange()))
        # we need to shuffle to randomize ties resolution
        numpy.random.shuffle(rates)
        winners = rates[0:k]
        heapq.heapify(winners)
        for r in rates[k:]:
            if r[0] > winners[0][0]:
                heapq.heapreplace(winners, r)
        argwinners = [(w[1] / rate_enc.shape[0], w[1] % rate_enc.shape[0])
                      for w in winners]
    return argwinners
