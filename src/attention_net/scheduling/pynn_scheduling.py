#!/usr/bin/env python2
""" Utilities and algorithms to integrate PyNN and SimPy
"""

import SimPy.Simulation as simpy
from common.utils import LOGGER

SIMPY_END_T = 0

def configure_scheduling():
    SimPy.Simulation.initialize()

def run_simulation():
    # WHOAAAAAAAAAAAAAAAAAAA complex. need to make the parallel
    # between simpy's run and pyNN's run, because we want to run
    # exactly the right length between 2 events...
    SimPy.Simulation.simulate(nutil=SIMPY_END_T)

class InputPresentation(simpy.Process):
    def __init__(self, population, input_sample):
        Process.__init__(self)
        self.name = "Presentation of " + str(input_sample) + \
            " to " + population.label
        self.population = population
        self.input_sample = input_sample

     def ACTIONS(self):
         # Does ACTIONS() connects input and run, or does run only
         # connect? There are several approaches.
         # 1) go() installs the input and sets up an event to remove
         #    the input. The running itself is done elsewhere.
         # 2) Assuming there is a SimPy hook for end-of event, we
         #    associate this hook with removal of the input. The
         #    running is done elsewhere and pynn() runs are atomically
         #    done between all events.
         # 3) go() installs the input, and does part of the pynn run
         #    for an atomically small duration in function of the next
         #    event, yields before the next event, and repeats that
         #    until the total duration of the presentation is
         #    reached. The end of input event is also installed here.
         # The decision to implement 1, 2, or 3 or any combination
         # depends on the facilities offered by simpy.
         LOGGER.info("%s starting", self.name)
         yield hold,self,100.0
         print now( ), self.name, "Arrived"

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
    
