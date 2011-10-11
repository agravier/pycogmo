#!/usr/bin/env python2

import itertools
import logging
from logging import NullHandler 
from mock import Mock, patch
from nose import with_setup
from nose.tools import eq_, raises, timed, nottest
from ui.graphical.visualisation import *
from ui.graphical.visualisation import VisualisableNetworkStructure as VNS

DUMMY_LOGGER = logging.getLogger("testLogger")
DUMMY_LOGGER.addHandler(NullHandler())
V = None

# holder class ("namespace") fot the test variables
class Tns(object):
    pass

def setup_vns():
    global V
    V = VNS(DUMMY_LOGGER)

def setup_units():
    Tns.l_id, Tns.l_x, Tns.l_y, Tns.l_z = xrange(5, 20), xrange(3, 18), \
        xrange(0,30,2), list(itertools.chain([None], itertools.repeat(-1,14)))
    Tns.vns_units = [VNS.Unit(u_id, x, y, z) 
                     for (u_id, x, y, z) in itertools.izip(Tns.l_id, Tns.l_x, Tns.l_y, Tns.l_z)]

##################################################
# Testing the VisualisableNetworkStructure class #
##################################################

### template
@with_setup(setup_units)
@with_setup(setup_vns)
def test_VNS_():
    assert False
nottest(test_VNS_)
### template

@with_setup(setup_units)
@with_setup(setup_vns)
def test_VNS_add_unit():
    "add_unit is complete and correct."
    assert len(V.units) == 0
    i = 0
    for u in Tns.vns_units:
        V.add_unit(u)
        u = V.units[i]
        assert (u.unit_id, u.x, u.y, u.z) == (Tns.l_id[i], Tns.l_x[i], Tns.l_y[i], Tns.l_z[i])
        i += 1
    assert len(V.units) == len(Tns.vns_units)
    V.assign_unit_to_map = Mock()
    V.add_unit(VNS.Unit(-1, 1, 2), "bar")
    assert V.assign_unit_to_map.called

@raises(VNS.UnitNotFoundError)
@with_setup(setup_units)
@with_setup(setup_vns)
def test_VNS_assign_unit_to_map_raises_():
    "assign_unit_to_map raises an exception on inexistent unit."
    V.assign_unit_to_map(Tns.vns_units[0], "foo")

@with_setup(setup_units)
@with_setup(setup_vns)
def test_VNS_assign_unit_to_map():
    "The maps of unit names get filled."
    V.add_unit(Tns.vns_units[0], "bar")
    V.add_unit(Tns.vns_units[1], "bar")
    assert V.maps["bar"] == [Tns.vns_units[0], Tns.vns_units[1]]

@with_setup(setup_units)
@with_setup(setup_vns)
def test_VNS_add_population():
    "add_population completeness and correctness."
    V.add_population(iter(Tns.vns_units))
    assert set(V.units) == set(Tns.vns_units)

@raises(VNS.UnitNotFoundError)
@with_setup(setup_units)
@with_setup(setup_vns)
def test_VNS_connect_units_inexistent_id():
    "connect_units with inexistent units raises UnitNotFoundError."
    V.connect_units(Tns.vns_units[0].unit_id, Tns.vns_units[1].unit_id, 1)

@raises(VNS.UnitNotFoundError)
@with_setup(setup_units)
@with_setup(setup_vns)
def test_VNS_connect_units_inexistent_unit():
    "connect_units with inexistent units raises UnitNotFoundError."
    V.connect_units(Tns.vns_units[0], Tns.vns_units[1], 1)


@raises(VNS.WeightOutOfRangeError)
@with_setup(setup_units)
@with_setup(setup_vns)
def test_VNS_connect_units_wrong_weight():
    "connect_units with incorrect weight raises WeightOutOfRangeError."
    V.add_population(iter(Tns.vns_units))
    V.connect_units(Tns.vns_units[0], Tns.vns_units[1], 2)

@with_setup(setup_units)
@with_setup(setup_vns)
def test_VNS_connect_units():
    "connect_units correctly modifies units_conn."
    V.add_population(iter(Tns.vns_units))
    V.connect_units(Tns.vns_units[0], Tns.vns_units[1], -1)
    V.connect_units(Tns.vns_units[1], Tns.vns_units[0], 0.3)
    id0, id1 = Tns.vns_units[0].unit_id, Tns.vns_units[1].unit_id
    assert set(V.units_conn) == set([(id0, id1, -1), (id1, id0, 0.3)])

@with_setup(setup_units)
@with_setup(setup_vns)
def test_VNS_connect_units_list():
    "connect_units_list correctly modifies units_conn."
    V.connect_units = Mock()
    V.add_population(iter(Tns.vns_units))
    c_l = itertools.izip([Tns.l_id, reversed(Tns.l_id), itertools.repeat(1)])
    V.connect_units_list(c_l)
    assert set(c_l) == set(V.units_conn)

@with_setup(setup_units)
@with_setup(setup_vns)
def test_VNS_connect_maps():
    "connect_maps correctly modifies maps_conn."
    V.add_unit(VNS.Unit(1, 1, 2), "bar")
    V.add_unit(VNS.Unit(2, 3, 2), "foo")
    V.connect_maps("bar", "foo")
    assert V.maps_conn == [("bar", "foo")]

###########################
# general setup functions #
###########################

@with_setup()
def test_vtkTimerCallback():
    assert False

@with_setup()
def test_visualisation_process_f():
    assert False

@with_setup()
def test_setup_visualisation():
    assert False

@with_setup()
def test_map_source_object():
    assert False

@with_setup()
def test_add_actors_to_scene():
    assert False

@with_setup()
def test_prepare_render_env():
    assert False

@with_setup()
def test_setup_timer():
    assert False

#######################################
# Network structure drawing functions #
#######################################

# Template
@with_setup()
def test_():
    assert False
nottest(test_)
# Template

