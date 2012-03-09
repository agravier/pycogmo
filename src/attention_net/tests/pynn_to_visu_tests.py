#!/usr/bin/env python2

import itertools
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


# holder class ("namespace") for the test variables
class Tns(object):
    pass


def setup_and_fill_adapter():
    setup_adapter()
    Tns.pop_size = 27
    Tns.pynn_pop1 = pynnn.Population(Tns.pop_size, pynnn.IF_cond_alpha)
    Tns.ids1 = [int(u) for u in Tns.pynn_pop1.all()]
    Tns.pynn_pop2 = pynnn.Population(Tns.pop_size, pynnn.IF_cond_alpha,
                                 structure=pynnn.space.Grid3D())
    Tns.ids2 = [int(u) for u in Tns.pynn_pop2.all()]
    A.add_pynn_population(Tns.pynn_pop1)
    Tns.pop2_alias = "testmap"
    A.add_pynn_population(Tns.pynn_pop2, alias=Tns.pop2_alias)
    Tns.pynn_proj1 = pynnn.Projection(Tns.pynn_pop1, Tns.pynn_pop2,
                                  pynnn.OneToOneConnector())
    Tns.pynn_proj2 = pynnn.Projection(Tns.pynn_pop2, Tns.pynn_pop1,
                                  pynnn.AllToAllConnector())
    A.add_pynn_projection(Tns.pynn_pop1, Tns.pynn_pop2,
                          Tns.pynn_proj1.connection_manager)
    A.add_pynn_projection(Tns.pynn_pop2, Tns.pynn_pop1,
                          Tns.pynn_proj2.connection_manager)


@with_setup(setup_adapter)
def test_adapter_locked_states():
    "basic lock/unlock for changes adapter state test"
    assert A.check_open()
    A.commit_structure()
    assert not A.check_open()


@with_setup(setup_adapter)
def test_adapter_methods_call_check_open():
    """methods in the methods_checking_open list have called check_open"""
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
    """add_pynn_population checks the int value of each unit it's given."""
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
    A.add_pynn_population(pynn_pop2, alias = "testmap")
    for u in pynn_pop2.all():
        assert u.__int__.call_count == pop_size, "units missed in the 3D case"


@with_setup(setup_and_fill_adapter)
def test_add_pynn_population_sets_up_labels_and_aliases():
    pynn_pop3 =  pynnn.Population(1, pynnn.IF_cond_alpha)
    A.add_pynn_population(pynn_pop3)
    assert A.aliases[Tns.pynn_pop1.label] == Tns.pynn_pop1.label
    assert A.aliases[Tns.pynn_pop2.label] == Tns.pop2_alias
    assert A.aliases[pynn_pop3.label] == pynn_pop3.label


@with_setup(setup_adapter)
def test_adapter_keeps_unit_count():
    """Add_pynn_population and commit_structure result in consistent number of
    units."""
    assert A.num_units == 0
    pop_size = 27
    pynn_pop1 = pynnn.Population(pop_size, pynnn.IF_cond_alpha)
    A.add_pynn_population(pynn_pop1, alias = "soilwork")
    pynn_pop2 = pynnn.Population(pop_size, pynnn.IF_cond_alpha,
                                 structure = pynnn.space.Grid3D())
    A.add_pynn_population(pynn_pop2)
    A.commit_structure()
    assert A.num_units == pop_size * 2


@with_setup(setup_and_fill_adapter)
def test_add_pynn_projection_adds_all_connections():
    """Tests if add_pynn_projection adds exactly the connections it's
    given to its internal units_connections list."""
    for c in itertools.groupby(A.units_connections, key=lambda x:x[0]):
        out_it = c[1] # iterator on outbounds cx from unit c[0]
        if c[0] in Tns.ids1:
            assert out_it.next()[1] in Tns.ids2
            try:
                out_it.next()
                assert False, ("There should only be one outbound connection"
                               " from this unit.")
            except StopIteration:
                pass
        elif c[0] in Tns.ids2:
            o_l = [o[1] for o in out_it]
            assert set(o_l) == set(Tns.ids1)
        else:
            assert False, "Unit ID inexistent on the PyNN side."


@with_setup(setup_and_fill_adapter)
def test_commit_structure_results_in_complete_output_struct():
    """Tests the completeness of the output structure's units and
    connections."""
    # TODO: compare A.output_struct and a hand-made version. only doable when
    # VisualisableNEtworkStructure is done and tested.
    A.commit_structure()
    out_units = set([u.unit_id for u in A.output_struct.units])
    assert out_units == set(Tns.ids2 + Tns.ids1)
    out_maps_aliases = A.output_struct.maps
    loma = list(out_maps_aliases.iterkeys())
    assert len(loma) == 2
    assert "testmap" in loma
    out_units_conn = A.output_struct.units_conn
    # one to one in id1 -> id2, all to all in id2 -> id1
    out_conn = set([(s, r) for (s, r, _) in out_units_conn])
    print "IDS1", Tns.ids1
    print "IDS2", Tns.ids2
    print "out_u_conn", out_units_conn
    print "out_conn", out_conn
    for s, r in itertools.izip(Tns.ids1, Tns.ids2):
        print (s, r)
        assert (s, r) in out_conn
    for s, r in itertools.product(Tns.ids2, Tns.ids1):
        assert (s, r) in out_conn
    assert len(out_conn) == Tns.pop_size + Tns.pop_size**2
    out_maps_conn = A.output_struct.maps_conn
    assert len(out_maps_conn) ==  2
