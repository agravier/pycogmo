#!/usr/bin/env python2
""" Implements algorithms that schedule the training of the PyNN
network.
"""

from SimPy.Simulation import Process

from common.pynn_utils import get_weights, set_weights, get_input_layer
from common.utils import log_tick, LOGGER
from scheduling.pynn_scheduling import run_simulation, \
    schedule_input_presentation, schedule_output_rate_calculation
import numpy

# epoch: 1 series of presentations of the whole set
# presentation: presentation of one sample during a certain number of
# cycles or a certain simulated duration
# cycle: 1 atomic timestep in the network simulator


def _make_so_learning_rule(description, lambda_expr):
    """"Returns a function that implements the learning algorithm given in
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


def train_kwta(population,
               input_samples,
               num_winners,
               neighbourhood_fn,
               presentation_duration,
               stop_condition):
    """Self-organized learning. The population should have k-WTA compatible
    lateral inhibition. The neighbourhood function is used for neighbourhood
    learning only, not for inhibition."""
    while not stop_condition:
        kwta_epoch(population, input_samples, presentation_duration)
    

def kwta_epoch(population, samples, duration):
    for s in samples:
        kwta_presentation(population, s, duration)


def kwta_presentation(population, sample, duration):
    schedule_input_presentation(population, sample, duration)
    schedule_output_rate_calculation(population, duration)
    run_simulation()
    
