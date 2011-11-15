#!/usr/bin/env python2

from nose import with_setup
from scheduling.pynn_scheduling import *

class DummyProcess(sim.Process):
    def ACTIONS(self):
        yield hold,self,1

def setup_clean_simpy():
    global SIMPY_END_T
    SIMPY_END_T = -1
    configure_scheduling()

