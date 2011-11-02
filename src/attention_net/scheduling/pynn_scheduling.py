#!/usr/bin/env python2
""" Utilities and algorithms to integrate PyNN and SimPy
"""

import SimPy.Simulation

SIMPY_END_T = 0

def configure_scheduling():
    SimPy.Simulation.initialize()

def run_simulation():
    SimPy.Simulation.simulate(until=SIMPY_END_T)

