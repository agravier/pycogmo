#!/usr/bin/env python2

import itertools
import logging
from logging import NullHandler 
from mock import Mock, patch
import multiprocessing
from nose import with_setup
from nose.tools import eq_, raises, timed, nottest
import ui.graphical.visualisation
from ui.graphical.visualisation import *
from ui.graphical.visualisation import VisualisableNetworkStructure as VNS
import vtk

multiprocessing.get_logger().addHandler(NullHandler())
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
def test_VNS___eq__():
    "Equality of two VisualisableNetworkStructures."
    u1, u2 = VNS.Unit(1, 1, 2), VNS.Unit(2, 3, 2)
    V.add_population(iter(Tns.vns_units))
    V.connect_units(Tns.vns_units[0], Tns.vns_units[1], -1)
    V.connect_units(Tns.vns_units[1], Tns.vns_units[0], 0.3)
    V.add_unit(u1, "bar")
    V.add_unit(u2, "foo")
    V.connect_maps("bar", "foo")
    w = VNS(DUMMY_LOGGER)
    w.add_unit(u2, "foo")
    w.add_population(iter(Tns.vns_units))
    w.add_unit(u1, "bar")
    w.connect_units(Tns.vns_units[0], Tns.vns_units[1], -1)
    w.connect_units(Tns.vns_units[1], Tns.vns_units[0], 0.3)
    w.connect_maps("bar", "foo")
    assert V == w

@with_setup(setup_units)
@with_setup(setup_vns)
def test_VNS_not__eq__():
    "Inequality of two VisualisableNetworkStructures."
    u1, u2 = VNS.Unit(1, 1, 2), VNS.Unit(2, 3, 2)
    V.add_population(iter(Tns.vns_units))
    V.connect_units(Tns.vns_units[0], Tns.vns_units[1], -1)
    V.connect_units(Tns.vns_units[1], Tns.vns_units[0], 0.3)
    V.add_unit(u1, "bar")
    V.add_unit(u2, "foo")
    V.connect_maps("bar", "foo")
    w = VNS(DUMMY_LOGGER)
    w.add_unit(u2, "foo")
    w.add_population(iter(Tns.vns_units))
    w.add_unit(u1, "bar")
    w.connect_units(Tns.vns_units[0], Tns.vns_units[1], 1) # changed
    w.connect_units(Tns.vns_units[1], Tns.vns_units[0], 0.3)
    w.connect_maps("bar", "foo") 
    x = VNS(DUMMY_LOGGER)
    x.add_unit(u2, "foo")
    x.add_population(iter(Tns.vns_units))
    x.add_unit(u1, "lol") # changed
    x.connect_units(Tns.vns_units[0], Tns.vns_units[1], -1)
    x.connect_units(Tns.vns_units[1], Tns.vns_units[0], 0.3)
    x.connect_maps("bar", "foo")
    y = VNS(DUMMY_LOGGER)
    y.add_unit(u2, "foo")
    y.add_population(iter(Tns.vns_units))
    y.add_unit(u1, "bar")
    # y.connect_units(Tns.vns_units[0], Tns.vns_units[1], -1) changed
    y.connect_units(Tns.vns_units[1], Tns.vns_units[0], 0.3)
    y.connect_maps("bar", "foo")
    z = VNS(DUMMY_LOGGER)
    z.add_unit(u2, "foo")
    z.add_population(iter(Tns.vns_units))
    z.add_unit(u1, "bar")
    z.connect_units(Tns.vns_units[0], Tns.vns_units[1], -1)
    z.connect_units(Tns.vns_units[1], Tns.vns_units[0], 0.3)
    z.connect_maps("bar", "faz") # changed
    assert V != w
    assert V != x
    assert V != y
    assert V != z

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
    "The mapping of units assignments to maps gets filled."
    V.add_unit(Tns.vns_units[0], "bar")
    V.add_unit(Tns.vns_units[1], "bar")
    assert V.maps["bar"] == [Tns.vns_units[0].unit_id, Tns.vns_units[1].unit_id]

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

def setup_child_conn_and_callback():
    Tns.parent_conn, Tns.child_conn = multiprocessing.Pipe()
    Tns.vtc = vtkTimerCallback(Tns.child_conn, Mock())

def setup_patch_visualisation_functions():
    Tns.setup_vis_patch = patch("ui.graphical.visualisation.setup_visualisation")
    Tns.add_actors_patch = patch("ui.graphical.visualisation.add_actors_to_scene")
    Tns.prepare_env_patch = patch("ui.graphical.visualisation.prepare_render_env")
    Tns.setup_timer_patch = patch("ui.graphical.visualisation.setup_timer")
    Tns.setup_vis_mock = Tns.setup_vis_patch.start()
    Tns.setup_vis_mock.return_value = (Mock(), Mock(), Mock())
    Tns.add_actors_mock = Tns.add_actors_patch.start()
    Tns.prepare_env_mock = Tns.prepare_env_patch.start()
    Tns.setup_timer_mock = Tns.setup_timer_patch.start()

def teardown_patch_visualisation_functions():
    Tns.setup_vis_patch.stop()
    Tns.setup_vis_mock = None
    Tns.add_actors_patch.stop()
    Tns.add_actors_mock = None
    Tns.prepare_env_patch.stop()
    Tns.prepare_env_mock = None
    Tns.setup_timer_patch.stop()
    Tns.setup_timer_mock = None

@timed(1)
@with_setup(setup_child_conn_and_callback)
def test_vtkTimerCallback_execute_does_not_block():
    "vtkTimerCallback.execute doesn't block on empty pipe."
    p = multiprocessing.Process(
        target=Tns.vtc.execute, args=(Mock(), Mock()))
    p.start()
    p.join(2)

@with_setup(setup_patch_visualisation_functions, teardown_patch_visualisation_functions)
def test_visualisation_process_f_side_effects():
    """The process initialization function calls all setup functions."""
    global LOGGER
    mock_pipe, mock_logger = Mock(), Mock()
    visualisation_process_f(mock_pipe, mock_logger)
    ui.graphical.visualisation.add_actors_to_scene = Mock()
    ui.graphical.visualisation.prepare_render_env = Mock()
    ui.graphical.visualisation.setup_timer = Mock(return_value=1)
    assert Tns.setup_vis_mock.called
    assert ui.graphical.visualisation.LOGGER is mock_logger
    assert Tns.add_actors_mock.called
    assert Tns.prepare_env_mock.called
    assert Tns.setup_timer_mock.called

def test_setup_visualisation():
   """Return values are renderer, window and interactor linked
   together."""
   r, w, i = setup_visualisation()
   assert isinstance(r, vtk.vtkRenderer)
   assert isinstance(w, vtk.vtkRenderWindow)
   assert isinstance(i, vtk.vtkRenderWindowInteractor)
   assert r is w.GetRenderers().GetFirstRenderer()
   assert w is i.GetRenderWindow()

def test_map_source_object():
    "Returns actor -> mapper -> object."
    obj = vtk.vtkPointSource()
    m, a = map_source_object(obj)
    assert isinstance(m.GetInput(), vtk.vtkObject)
    assert m is a.GetMapper()

def test_add_actors_to_scene():
    "add_actors_to_scene calls renderer.AddActor for all actors."
    r = Mock()
    r.AddActor = Mock(spec=vtk.vtkRenderer.AddActor)
    add_actors_to_scene(r, 1)
    r.AddActor.assert_called_with(1)
    r = Mock()
    add_actors_to_scene(r, 2, 3, 4)
    assert r.method_calls == [("AddActor", (2,)), ("AddActor", (3,)),
                              ("AddActor", (4,))]

def test_prepare_render_env():
    "prepare_render_env calls interactor.Initialize and win.Render."
    m = Mock()
    prepare_render_env(m, m)
    assert m.Initialize.called
    assert m.Render.called

def test_setup_timer():
    vtkTimerCallback_patch = patch("ui.graphical.visualisation.vtkTimerCallback")
    vtkTimerCallback_mock = vtkTimerCallback_patch.start()
    m = Mock()
    vtkTimerCallback_mock.return_value = m
    setup_timer(m, 1, 2)
    vtkTimerCallback_patch.stop()
    assert vtkTimerCallback_mock.call_args_list == [((1,2),{})]
    assert m.method_calls[0] == ("AddObserver", 
                                 ("TimerEvent", m.execute), {})
    assert m.method_calls[1][0] == "CreateRepeatingTimer"

#######################################
# Network structure drawing functions #
#######################################

# Template
@with_setup()
def test_():
    assert False
nottest(test_)
# Template

