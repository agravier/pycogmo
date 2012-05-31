#!/usr/bin/env python2

# Copyright 2011, 2012 Alexandre Gravier (al.gravier@gmail)

# This file is part of PyCogMo.
# PyCogMo is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# PyCogMo is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with PyCogMo.  If not, see <http://www.gnu.org/licenses/>.

""" This is a sandbox file that I use for development.
"""

import pyNN.brian as pynnn
# from pyNN.recording.files import HDF5ArrayFile needs cython and
# tables (=pain) 
from pyNN.utility import init_logging
import cPickle as pickle
import logging
import multiprocessing # because threading will not bypass the GIL
from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL, FATAL
from multiprocessing import SUBDEBUG, SUBWARNING
import sys
import time
# -- own modules
import common.utils as utils
from common.utils import log_tick, LOGGER, make_logfile_name, ensure_dir
import ui.graphical.visualisation as visualisation
import ui.graphical.pynn_to_visu as pynn_to_visu

# Send an activity update to the visualisation process every
# SIMU_TO_VISU_MESSAGE_PERIOD of simulated time.
SIMU_TO_VISU_MESSAGE_PERIOD = 100

# Total simulated duration
SIMU_DURATION = 1000

SIMU_TIMESTEP = 0.1

VISU_PROCESS_JOIN_TIMEOUT = 10


def setup_populations_recording(p, *args):
    """calls record(to_file=False), record_gsyn(to_file=False),
    record_v(to_file=False) on the populations in argument"""
    for pop in (p,) + args:
        pop.record(to_file=False)
        pop.record_gsyn(to_file=False)
        pop.record_v(to_file=False)

# parent->child test
def ipc_test(parent_conn, child_conn):
    for a in reversed(range(-1,100)):
        time.sleep(0.1)
        log_tick("just before send")
        # Only pipe in data to be visualised if visualisation pipe is
        # empty.
        if (not child_conn.poll()):
            log_tick("the vis. pipeline is empty, putting in some data")
            parent_conn.send(a)
        log_tick("just after send")

########
# Main #
########

def main():
    ## Uninteresting setup, start up the visu process,...
    logfile = make_logfile_name()
    ensure_dir(logfile)
    f_h = logging.FileHandler(logfile)
    f_h.setLevel(SUBDEBUG)
    d_h = logging.StreamHandler()
    d_h.setLevel(INFO)
    utils.configure_loggers(debug_handler=d_h, file_handler=f_h)
    parent_conn, child_conn = multiprocessing.Pipe()
    p = multiprocessing.Process(
        target=visualisation.visualisation_process_f,
        name="display_process", args=(child_conn, LOGGER))
    p.start()

    pynnn.setup(timestep=SIMU_TIMESTEP)
    init_logging("logfile", debug=True)
    LOGGER.info("Simulation started with command: %s", sys.argv)

    ## Network setup
    # First population
    p1 = pynnn.Population(100, pynnn.IF_curr_alpha,
                          structure=pynnn.space.Grid2D())
    p1.set({'tau_m':20, 'v_rest':-65})
    # Second population
    p2 = pynnn.Population(20, pynnn.IF_curr_alpha,
                          cellparams={'tau_m': 15.0, 'cm': 0.9})
    # Projection 1 -> 2
    prj1_2 = pynnn.Projection(
        p1, p2, pynnn.AllToAllConnector(allow_self_connections=False),
        target='excitatory')
    # I may need to make own PyNN Connector class. Otherwise, this is
    # neat:  exponentially decaying probability of connections depends
    # on distance. Distance is only calculated using x and y, which
    # are on a toroidal topo with boundaries at 0 and 500.
    connector = pynnn.DistanceDependentProbabilityConnector(
        "exp(-abs(d))",
        space=pynnn.Space(
            axes='xy', periodic_boundaries=((0,500), (0,500), None)))
    # Alternately, the powerful connection set algebra (python CSA
    # module) can be used.
    weight_distr = pynnn.RandomDistribution(distribution='gamma',
                                            parameters=[1,0.1])
    prj1_2.randomizeWeights(weight_distr)

    # This one is in NEST but not in Brian:
    # source = pynnn.NoisyCurrentSource(
    #     mean=100, stdev=50, dt=SIMU_TIMESTEP, 
    #     start=10.0, stop=SIMU_DURATION, rng=pynnn.NativeRNG(seed=100)) 
    source = pynnn.DCSource(
        start=10.0, stop=SIMU_DURATION, amplitude=100) 
    source.inject_into(list(p1.sample(50).all()))

    p1.record(to_file=False)
    p2.record(to_file=False)

    ## Build and send the visualizable network structure
    adapter = pynn_to_visu.PynnToVisuAdapter(LOGGER)
    adapter.add_pynn_population(p1)
    adapter.add_pynn_population(p2)
    adapter.add_pynn_projection(p1, p2, prj1_2.connection_manager)
    adapter.commit_structure()
    
    parent_conn.send(adapter.output_struct)
    
    # Number of chunks to run the simulation:
    n_chunks = SIMU_DURATION // SIMU_TO_VISU_MESSAGE_PERIOD
    last_chunk_duration = SIMU_DURATION % SIMU_TO_VISU_MESSAGE_PERIOD
    # Run the simulator
    for visu_i in xrange(n_chunks):
        pynnn.run(SIMU_TO_VISU_MESSAGE_PERIOD)
        parent_conn.send(adapter.make_activity_update_message())
        LOGGER.debug("real current p1 spike counts: %s",
                     p1.get_spike_counts().values())
    if last_chunk_duration > 0:
        pynnn.run(last_chunk_duration)
        parent_conn.send(adapter.make_activity_update_message())
    # Cleanup
    pynnn.end()
    # Wait for the visualisation process to terminate
    p.join(VISU_PROCESS_JOIN_TIMEOUT)
    


if __name__ == "__main__":
    main()
