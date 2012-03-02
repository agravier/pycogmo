#!/usr/bin/env python2

from mock import Mock, patch
from nose import with_setup
from nose.tools import eq_, raises, timed, nottest
import SimPy.Simulation as sim

from common.pynn_utils import InputSample, RectilinearInputLayer, \
    InvalidMatrixShapeError
from scheduling.pynn_scheduling import *

from tests.pynn_utils_tests import Tns, setup_weights, \
    setup_pynn_populations, setup_rectinilearinputlayers


FULL_BINARY_CHECKER = [[1, 0, 1, 0, 1, 0, 1, 0],
                       [0, 1, 0, 1, 0, 1, 0, 1],
                       [1, 0, 1, 0, 1, 0, 1, 0],
                       [0, 1, 0, 1, 0, 1, 0, 1],
                       [1, 0, 1, 0, 1, 0, 1, 0],
                       [0, 1, 0, 1, 0, 1, 0, 1],
                       [1, 0, 1, 0, 1, 0, 1, 0],
                       [0, 1, 0, 1, 0, 1, 0, 1]]

def setup_input_samples():
    Tns.s1 = InputSample(8, 8, FULL_BINARY_CHECKER)

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
    assert get_current_time() == sim.now()
    assert sim.now() == 0
    assert pynnn.get_current_time() == 0
    run_simulation(end_time = 10)
    assert sim.now() == 10
    assert pynnn.get_current_time() == 10 # relaxed cond  <= 10
    run_simulation(end_time = 20)
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
    assert get_current_time() == sim.now()
    assert pynnn.get_current_time() == 205
    run_simulation(end_time = 300)
    assert sim.now() == 300
    assert pynnn.get_current_time() == 300

@with_setup(setup_rectinilearinputlayers)
@with_setup(setup_input_samples)
def test_input_presentation_init_correct_shape():
    p = InputPresentation(Tns.ril1, Tns.s1, 1)
    assert p.duration == 1

@with_setup(setup_rectinilearinputlayers)
@raises(InvalidMatrixShapeError)
def test_input_presentation_init_incorrect_shape1():
    p = InputPresentation(Tns.ril1, InputSample(7, 8, [[0]*8]*7), 1)

@with_setup(setup_rectinilearinputlayers)
@raises(InvalidMatrixShapeError)
def test_input_presentation_init_incorrect_shape2():
    p = InputPresentation(Tns.ril1, InputSample(8, 7, [[0]*7]*8), 1)

def mock_input_layer_apply_input_setup():
    Tns.il_ai_patcher = patch("common.pynn_utils.RectilinearInputLayer.apply_input")
    Tns.il_ai_mock = Tns.il_ai_patcher.start()

def mock_input_layer_apply_input_teardown():
    Tns.il_ai_patcher.stop()
    del Tns.il_ai_mock

def setup_samples_layers_sim():
    setup_rectinilearinputlayers()
    setup_input_samples()
    setup_clean_simpy()

@with_setup(mock_input_layer_apply_input_setup, mock_input_layer_apply_input_teardown)
@with_setup(setup_samples_layers_sim)
def test_input_presentation_actions():
    p = InputPresentation(Tns.ril1, Tns.s1, 1)
    p.ACTIONS().next()
    now = sim.now()
    Tns.il_ai_mock.assert_called_once_with(Tns.s1, now, 1)

@with_setup(setup_samples_layers_sim)
def test_schedule_input_presentation():
    schedule_input_presentation(Tns.p1,
                                Tns.s1, 
                                10)
    assert sim.peek() == sim.now()
    schedule_input_presentation(Tns.p1,
                                Tns.s1, 
                                10,
                                start_t = 20)
    assert sim.Globals.allEventTimes() == [0,20]
    from scheduling.pynn_scheduling import SIMPY_END_T as end_t
    assert end_t == 30

