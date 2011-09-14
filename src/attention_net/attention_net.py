#!/usr/bin/env python2.7

import pyNN.nest as pynnn
# from pyNN.recording.files import HDF5ArrayFile needs cython and
# tables (=pain) 
from pyNN.utility import init_logging
import logging
import multiprocessing # because threading will not bypass the GIL
import sys
import time
# -- own modules
import utils
from utils import log_tick, LOGGER
import visualisation
import pynn_to_visu

def setup_populations_recording(p, *args):
    """calls record(to_file=False), record_gsyn(to_file=False),
    record_v(to_file=False) on the populations in argument"""
    for pop in (p,) + args:
        pop.record(to_file=False)
        pop.record_gsyn(to_file=False)
        pop.record_v(to_file=False)
        
def main():
    utils.configure_loggers()
    parent_conn, child_conn = multiprocessing.Pipe()
    p = multiprocessing.Process(
        target=visualisation.visualisation_process_f,
        name="display_process", args=(child_conn, LOGGER))
    p.start()

    # parent->child test
    for a in reversed(range(-1,100)):
        time.sleep(0.1)
        log_tick("just before send")
        # Only pipe in data to be visualised if visualisation pipe is
        # empty.
        if (not child_conn.poll()):
            log_tick("the vis. pipeline is empty, putting in some data")
            parent_conn.send(a)
        log_tick("just after send")
    p.join()

    pynnn.setup()
    init_logging("logfile", debug=True)
    print sys.argv[0]
    p1 = pynnn.Population(100, pynnn.IF_curr_alpha,
                          structure=pynnn.space.Grid2D())

    vis_struct = None
    
    p1.set({'tau_m':20, 'v_rest':-65})
    p2 = pynnn.Population(20, pynnn.IF_curr_alpha,
                          cellparams={'tau_m': 15.0, 'cm': 0.9})
    prj1_2 = pynnn.Projection(
        p1, p2, pynnn.AllToAllConnector(allow_self_connections=False),
        target='excitatory')
    # I may need to make own PyNN Connector class. Otherwise, this is
    # neat:  exponentially decaying probability of connections depends
    # on distance. Distance is only calculated using x and y, which
    # are on a toroidal topo with boundaries at 0 and 500.
    connector = pynnn.DistanceDependentProbabilityConnector("exp(-abs(d))",
                   space=pynnn.Space(axes='xy',
                               periodic_boundaries=((0,500), (0,500), None)))
    # Otherwise, the very leet connection set algebra (python CSA
    # module) can be used.
    weight_distr = pynnn.RandomDistribution(distribution='gamma',
                                            parameters=[1,0.1])
    prj1_2.randomizeWeights(weight_distr)
    pynnn.run(100.0)
    pynnn.end()


if __name__ == "__main__":
    main()
