#!/usr/bin/env python2

from nose import with_setup
from scheduling.pynn_scheduling import *

class TestProcess(sim.Process):
    def __init__(self, duration=0):
        sim.Process.__init__(self, name="Dummy (unit testing)")
        self.d = duration
    def ACTIONS(self):
        yield sim.hold,self,self.d

def setup_clean_simpy():
    global SIMPY_END_T
    SIMPY_END_T = -1
    configure_scheduling()

def setup_test_process():
    setup_clean_simpy()
    for t in range(20, 101, 20):
        TestProcess().start(at=t)

@with_setup(setup_clean_simpy)
def test_run_simulation_no_event():
    run_simulation()
    assert sim.now() == 0
    assert pynnn.get_current_time() == 0.0
    assert sim.Globals.allEventTimes() == []
    run_simulation(end_time = 50)
    assert sim.now() == 50
    assert pynnn.get_current_time() == 50.0
    assert sim.Globals.allEventTimes() == []

@with_setup(setup_test_process)
def test_run_simulation():
    run_simulation(end_time = 0)
    assert sim.now() == 0
    assert pynnn.get_current_time() == 0
    run_simulation(end_time = 10)
    assert sim.now() == 10
    assert pynnn.get_current_time() == 10 # relaxed cond  <= 10
    run_simulation(end_time = 20)
    print ("curtime ",  pynnn.get_current_time(), sim.now())
    assert sim.now() == 20
    assert pynnn.get_current_time() == 20
    run_simulation(end_time = 70)
    assert sim.now() == 70
    assert pynnn.get_current_time() == 70 # 60 >= t <=70
    run_simulation()
    assert sim.now() == 100
    assert pynnn.get_current_time() == 100
    TestProcess(5).start(at=200)
    run_simulation(end_time = 202)
    assert sim.now() == 202
    assert pynnn.get_current_time() == 202
    run_simulation(end_time = 203)
    assert sim.now() == 203
    assert pynnn.get_current_time() == 203
    run_simulation()
    assert sim.now() == 205
    assert pynnn.get_current_time() == 205
    run_simulation(end_time = 300)
    assert sim.now() == 300
    assert pynnn.get_current_time() == 300
