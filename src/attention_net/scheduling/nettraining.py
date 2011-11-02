#!/usr/bin/env python2
""" Implements algorithms that schedule the training of the PyNN
network.
"""

from SimPy.Simulation import Process

from common.pynn_utils import get_weights, set_weights
from common.utils import log_tick, LOGGER



def self_org_learn(population, 
                   samples, 
                   presentation_cycles,
                   stop_condition):
    pass
