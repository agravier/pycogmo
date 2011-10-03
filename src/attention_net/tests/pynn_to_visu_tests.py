#!/usr/bin/env python2

import logging
from logging import NullHandler 
from mock import Mock, patch
from nose import with_setup
from nose.tools import eq_, raises, timed
from ui.graphical.pynn_to_visu import *
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
    #A.reopen()
    #assert A.check_open()

@with_setup(setup_adapter)
def test_adapter_methods_call_check_open():
    """methods in the methods_checking_open list are checked to have
    called check_open"""
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
        assert A.check_open.called, \
            m[0].__name__ + " does not call check_open."
        A.check_open.reset_mock()

PATCH = None

def setup_mock_unit_unit_id():
    global PATCH
    PATCH = patch.object(pynnn.simulator.ID, "__int__")
    PATCH.start()

def teardown_mock_unit_id():
    global PATCH
    PATCH.stop()
    PATCH = None

@with_setup(setup_mock_unit_unit_id, teardown_mock_unit_id)
@with_setup(setup_adapter)
def test_add_pynn_population_processes_all_units():
    """Verifies if add_pynn_population checks the id of each unit u
    that it's given by checking if it accesses u's int value."""
    pop_size = 27
    pynnn.simulator.ID.__int__.return_value = 1
    pynn_pop1 = pynnn.Population(pop_size, pynnn.IF_cond_alpha)
    A.add_pynn_population(pynn_pop1)
    for u in pynn_pop1.all():
        assert u.__int__.call_count == pop_size, \
            "units missed in the 2D case"
    pynnn.simulator.ID.__int__.reset_mock()
    pynnn.simulator.ID.__int__.return_value = 1
    pynn_pop2 = pynnn.Population(pop_size, pynnn.IF_cond_alpha,
                                 structure = pynnn.space.Grid3D())
    A.add_pynn_population(pynn_pop2, concept_map = "testmap")
    for u in pynn_pop2.all():
        assert u.__int__.call_count == pop_size, "units missed in the 3D case"

@with_setup(setup_adapter)
def test_adapter_keeps_unit_count():
    """Verifies if several add_pynn_population followed by
    commit_structure leave the adapter in a consistent state regarding
    the number of units."""
    assert A.num_units == 0
    pop_size = 27
    pynn_pop1 = pynnn.Population(pop_size, pynnn.IF_cond_alpha)
    A.add_pynn_population(pynn_pop1, concept_map = "soilwork")
    pynn_pop2 = pynnn.Population(pop_size, pynnn.IF_cond_alpha,
                                 structure = pynnn.space.Grid3D())
    A.add_pynn_population(pynn_pop2)
    A.commit_structure()
    assert A.num_units == pop_size * 2

def test_add_pynn_projection_adds_all_connections():
    """Tests if add_pynn_projection adds exactly the connections it's
    given to its internal units_connections list"""
    assert False
