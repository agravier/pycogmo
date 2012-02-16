#!/usr/bin/env python2
""" Implements algorithms that schedule the training of the PyNN
network.
"""

from SimPy.Simulation import Process

from common.pynn_utils import get_weights, set_weights, get_input_layer
from common.utils import log_tick, LOGGER

# epoch: 1 series of presentations of the whole set
# presentation: presentation of one sample during a certain number of
# cycles or a certain simulated duration
# cycle: 1 atomic timestep in the network simulator

def hebb_learning(pre_syn_out, post_syn_act, weight, l_rate):
    "Hebbian learning"
    return weights + l_rate * post_syn_act * pre_syn_out

def oja_learning(pre_syn_out, post_syn_act, weight, l_rate):
    "Oja 1982, http://www.scholarpedia.org/article/Oja_learning_rule"
    return weight + l_rate * (post_syn_act * pre_syn_out - \
                              (post_syn_act**2) * weight)

def conditional_pca_learning(pre_syn_out, post_syn_act, weight, l_rate):
    "O'Reilly 2000, CECN"
    return weight + l_rate * post_syn_act * (pre_syn_out - weight)

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
    
