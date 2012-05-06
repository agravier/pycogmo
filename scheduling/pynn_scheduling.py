#!/usr/bin/env python2
""" Utilities and algorithms to integrate PyNN and SimPy
"""

import pyNN.brian as pynnn
import SimPy.Simulation as sim
from common.pynn_utils import get_input_layer, get_rate_encoder, \
    InputSample, RectilinearInputLayer, InvalidMatrixShapeError
from common.utils import LOGGER, optimal_rounding

SIMULATION_END_T = 0

pynnn.setup()
PYNN_TIME_STEP = pynnn.get_time_step()
PYNN_TIME_ROUNDING = optimal_rounding(PYNN_TIME_STEP)


class DummyProcess(sim.Process):
    def ACTIONS(self):
        yield sim.hold, self, 0


def get_current_time():
    return sim.now()


def configure_scheduling():
    global SIMULATION_END_T
    sim.initialize()
    pynnn.setup()
    SIMULATION_END_T = 0


def run_simulation(end_time=None):
    """Runs the simulation while keeping SimPy and PyNN synchronized at
    event times. Runs until no event is scheduled unless end_time is
    provided. if end_time is given, runs until end_time."""
    def run_pynn(end_t):
        pynn_now = pynnn.get_current_time()
        pynn_now_round = round(pynn_now, PYNN_TIME_ROUNDING)
        delta_t = round(end_t - pynn_now_round, PYNN_TIME_ROUNDING)
        if pynn_now <= pynn_now_round and delta_t > PYNN_TIME_STEP:
            delta_t = round(delta_t - PYNN_TIME_STEP, PYNN_TIME_ROUNDING)
        if delta_t > 0:  # necessary because run(0) may run PyNN by timestep
            pynnn.run(delta_t)  # neuralensemble.org/trac/PyNN/ticket/200
    is_not_end = None
    if end_time == None:
        # Would testing len(sim.Globals.allEventTimes()) be faster?
        is_not_end = lambda t: not isinstance(t, sim.Infinity)
    else:
        DummyProcess().start(at=end_time)
        is_not_end = lambda t: t <= end_time
    t_event_start = sim.peek()
    while is_not_end(t_event_start):
        LOGGER.debug("Progressing to SimPy event at time %s",
                     t_event_start)
        run_pynn(t_event_start) # run until event start
        sim.step() # process the event
        run_pynn(get_current_time()) # run PyNN until event end
        t_event_start = sim.peek()
        

class InputPresentation(sim.Process):
    def __init__(self, input_layer, input_sample, duration):
        """Process presenting input_sample (class
        common.pynn_utils.InputSample) to input_layer (class
        common.pynn_utils.RectilinearInputLayer) for duration ms."""
        sim.Process.__init__(self)
        self.name = "Presentation of " + str(input_sample) + \
            " to " + input_layer.pynn_population.label
        if input_layer.shape != input_sample.shape:
            raise InvalidMatrixShapeError(input_layer.shape[0],
                                          input_layer.shape[1],
                                          input_sample.shape[0],
                                          input_sample.shape[1])
        self.input_layer = input_layer
        self.input_sample = input_sample
        self.duration = duration
    
    def ACTIONS(self):
        LOGGER.debug("%s starting", self.name)
        self.input_layer.apply_input(self.input_sample, get_current_time(),
                                     self.duration)
        yield sim.hold, self, 0

DEFAULT_INPUT_PRESENTATION_DURATION = 200
def schedule_input_presentation(population,
                                input_sample,
                                start_t=None,
                                duration=DEFAULT_INPUT_PRESENTATION_DURATION):
    """Schedule the constant application of the input sample to the
    input layer, for duration ms, by default from start_t = current
    end of simulation, extending the simulation's scheduled end
    SIMULATION_END_T by the necessary number amount of time."""
    global SIMULATION_END_T
    input_layer = get_input_layer(population)
    if start_t == None:
        start_t = SIMULATION_END_T
    print "start_t is", start_t
    p = InputPresentation(input_layer, input_sample, duration)
    p.start(at=start_t)
    if start_t + duration > SIMULATION_END_T:
        SIMULATION_END_T = start_t + duration


class RateCalculation(sim.Process):
    """A RateCalculation process is a recurrent process initiated by the 
    schedule_input_presentation function. It handles the 
    RectilinearOutputRateEncoder object that it has been given at construction
    time by reading the time of the next update and scheduling the next
    RateCalculation."""
    # We don't want rate two calculation processes to extend the time of the
    # end of the simulation indefinitely by alternately modifying 
    # SIMULATION_END_T.
    # Hence, RateCalculation:
    #  - does not modify SIMULATION_END_T
    #  - only installs its next call if SIMULATION_END_T > sim.now()
    #  - installs its last call exactly at SIMULATION_END_T if no earlier 
    #    end_time was given.
    # The three conditions above are sufficient to ensure that exactly one last
    # call to each output rate encoder is performed at the end of the
    # simulation.
    def __init__(self, rate_encoder, end_t=None, correct_event_t=None):
        """expects a RectilinearOutputRateEncoder as parameter. end_t is the 
        time of the last recording to be scheduled. correct_event_t is a 
        workaround for the fact that pyNN.nest does not allow processing events
        at time 0. correct_event_t should be set when the current simulated time
        will be different than what it should be."""
        sim.Process.__init__(self)
        self._rate_encoder = rate_encoder
        self._end_t = end_t
        self.name = "Rate calculation for population " + \
            self._rate_encoder.pynn_population.label
        self._correct_event_t = correct_event_t

    @property
    def corrected_time(self):
        if self._correct_event_t == None:
            return get_current_time()
        return self._correct_event_t

    @property
    def last_schedulable_time(self):
        if self._end_t == None:
            return SIMULATION_END_T
        return min(self._end_t, SIMULATION_END_T)

    def ACTIONS(self):
        global SIMULATION_END_T
        LOGGER.debug("%s starting", self.name)
        self._rate_encoder.update_rates(get_current_time())
        if self.last_schedulable_time > self.corrected_time:
            # At least one more event has to be scheduled.
            next_period = self.corrected_time + self._rate_encoder.update_period
            next_time = min(next_period, self.last_schedulable_time)
            _schedule_output_rate_encoder(self._rate_encoder,
                                          start_t=next_time,
                                          end_t=self._end_t)
        yield sim.hold, self, 0


def schedule_output_rate_calculation(population, start_t=None, duration=None):
    """Schedules the recurrent calculation of the output rate of the given 
    population. A new RectilinearOutputRateEncoderis created with default 
    parameters if none is registered for this population. If no start_t is
    given, the current simulation time is used as start time. If no duration is
    given, the output rate encoder is active during the whole simulation."""
    rore = get_rate_encoder(population)
    if start_t == None:
        start_t = get_current_time()
    end_t = None
    if duration != None:
        end_t = duration + start_t
    _schedule_output_rate_encoder(rore, start_t, end_t)


def _schedule_output_rate_encoder(rate_enc, start_t, end_t):
    # workaround of the workaround of 
    # neuralensemble.org/trac/PyNN/ticket/200:
    rc = None
    if start_t == 0:
        rc = RateCalculation(rate_enc, end_t, correct_event_t=start_t)
        start_t += PYNN_TIME_STEP
    else:
        rc = RateCalculation(rate_enc, end_t)
    rc.start(at=start_t)
    print "events times", sim.Globals.allEventTimes()


