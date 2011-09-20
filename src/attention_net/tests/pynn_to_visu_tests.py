#!/usr/bin/env python2

import logging
from logging import NullHandler 
from mock import Mock 
from nose import with_setup
from nose.tools import eq_, raises, timed
from ..pynn_to_visu import *
import pyNN.nest as pynnn

DUMMY_LOGGER = logging.getLogger("testLogger")
DUMMY_LOGGER.addHandler(NullHandler())
A = None
pynnn.setup()

def setup_adapter():
    global A
    A = PynnToVisuAdapter(DUMMY_LOGGER)


@with_setup(setup_adapter)
def test_adapter_locked_states():
    "basic lock/unlock for changes adapter state test"
    assert A.check_open()
    A.commit_structure()
    assert not A.check_open()
    A.reopen()
    assert A.check_open()

@with_setup(setup_adapter)
def test_adapter_methods_call_check_open():
    A.check_open = Mock(return_value=True)
    pynn_pop1 = pynnn.Population(1, pynnn.IF_cond_alpha)
    pynn_pop2 = pynnn.Population(1, pynnn.IF_cond_alpha)
    pynn_prj = pynnn.Projection(
        pynn_pop1, pynn_pop2,
        pynnn.OneToOneConnector(),
        target='excitatory')
    pynn_cnx_mgr = pynn_prj.connection_manager
    pynn_u = pynn_pop1[0]
    methods_checking_open = [
        [A.assert_open, ()],
        [A.commit_structure, ()],
        [A.add_pynn_population, (pynn_pop1,)],
        [A.add_pynn_projection, (pynn_pop1, pynn_pop1,
                                     pynn_cnx_mgr)]]
    for m in methods_checking_open:
        m[0](*m[1])
        assert A.check_open.called
        A.check_open.reset()


