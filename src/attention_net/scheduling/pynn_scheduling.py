#!/usr/bin/env python2
""" Utilities and algorithms to integrate PyNN and SimPy
"""

import pyNN.nest as pynnn
import SimPy.Simulation as sim
from common.utils import LOGGER

SIMPY_END_T = -1

def configure_scheduling():
    sim.initialize()

def run_simulation(max_time = None):
    """Runs the simulation while keeping SimPy and PyNN synchronized at
    event times. Runs until no even is scheduled unless max_time is
    provided."""
    is_not_end = None
    if max_time == None:
        # Would testing len(sim.Globals.allEventTimes()) be faster?
        is_not_end = lambda t: return not isinstance(t, sin.Infinity)
    else:
        is_not_end = lambda t: return t < max_time
    t_next_event = sim.peek()
    while is_not_end(t_next_event):
        LOGGER.debug("Progressing to SimPy event at time %s",
                     t_next_event)
        pynnn.run(t_next_event - sim.now())
        sim.step()
        t_next_event = sim.peek()
        

class InputPresentation(sim.Process):
    def __init__(self, population, input_sample):
        Process.__init__(self)
        self.name = "Presentation of " + str(input_sample) + \
            " to " + population.label
        self.population = population
        self.input_sample = input_sample
    
    def ACTIONS(self):
        LOGGER.info("%s starting", self.name)
        yield sim.hold, self, 10.0
        print sim.now(), self.name, "Arrived"

def schedule_input_presentation(population, 
                                input_sample, 
                                duration,
                                start_t = None):
    """Schedule the constant application of the input sample to the
    population, for duration ms, by default from start_t = current end
    of simulation, extending the simulation's scheduled end
    SIMPY_END_T by duration milliseconds."""
    global SIMPY_END_T
    if start_t == None:
        start_t = SIMPY_END_T
    if start_t + duration > SIMPY_END_T:
        SIMPY_END_T = start_t + duration
    
