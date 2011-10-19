#!/usr/bin/env python2

import cPickle as picke
import datetime
import logging
import multiprocessing # because threading will not bypass the GIL
import os
import sys
import time
import vtk
# -- own modules
import common.utils as utils
from common.utils import log_tick

LOGGER = None # initialized by the child visualization process, should
              # be the muiltiprocessing module's logger because of
              # conccurent access.

NETWORK = None # The VisualisableNetwork that is a VTK source for
               # network display.

REN, REN_WIN, I_REN = None, None, None

################################
# VisualisableNetworkStructure #
################################

# Consiser using protocol buffers to pickle that if python's pickling
# causes problems
class VisualisableNetworkStructure(object):
    """Initial message sent to a visualiser child process to set up
    the physical elements in 3d space. This structure is abstract of
    particular visual choices, projections, transformations, or
    whether a neural map should be represented as a sheet, a cloud of
    units, or anything else. It's the visualisation interface that is
    to make or let the user make these choices."""
    
    class UnitNotFoundError(Exception):
        pass
    
    class WeightOutOfRangeError(Exception):
        pass
    
    class Unit(object):
        def __init__(self, unit_id, x, y, z=None):
            self.unit_id = unit_id
            self.x, self.y, self.z = x, y, z
        def __eq__(self, other):
            if isinstance(other, (int, long)):
                return self.unit_id == other
            e = self.unit_id == other.unit_id
            if e and (self.x != other.x or self.y != other.y or \
                      self.z != other.y):
                global LOGGER
                LOGGER.error("Units with ID %i and %i and coordinates %s and %s can't co-exist.",
                             self.unit_id, other.unit_id, 
                             (self.x, self.y, self.z), 
                             (other.x, other.y, other.z))
                return false
            return e
        def __int__(self):
            return self.unit_id
        @property
        def coords(self):
            if self.z is None:
                return (self.x, self.y)
            return (self.x, self.y, self.z)
    
    def __init__(self, logger):
        self.logger = logger
        # all units in consistent order, the same order that is to be
        # used when transmitting activity updates.
        self.units = list()
        # info about the conceptual grouping of units, a dict of
        # (name of the map, list of unit global indices)
        self.maps = dict()
        # detailed connectivity (unit-to-unit)
        self.units_conn = list()
        # abstract connectivity (between maps)
        self.maps_conn = list()

    def __eq__(self, other):
        """Structural comparison of the networks and comparison of the
        IDs of units and names of maps."""
        return (set(self.units) == set(other.units) 
                and self.maps == other.maps
                and set(self.units_conn) == set(other.units_conn)
                and set(self.maps_conn) == set(other.maps_conn))

    def add_unit(self, unit, assign_map = None):
        """appends a unit to the global list of units and to the
        assigned map's units list."""
        self.units.append(unit)
        self.assign_unit_to_map(unit, assign_map)

    def assign_unit_to_map(self, unit, assign_map):
        """Assigns the already existing unit of type
        VisualisableNetworkStructure.Unit to the given map that may
        need to be created for the occasion."""
        if unit not in self.units:
            raise self.UnitNotFoundError()
        if assign_map not in self.maps:
            self.maps[assign_map]=[]
        self.maps[assign_map].append(int(unit))
    
    def add_population(self, iterable_population, override_map = None):
        """appends a group of units to the list of units."""
        for u in iterable_population:
            self.add_unit(u, override_map)
    
    def connect_units(self, snd_unit_id, rcv_unit_id, strength):
        """appends a connection between two units (referenced by their
        id numbers) to the list of connections. Strength is between -1
        and 1"""
        snd_unit_id = int(snd_unit_id)
        rcv_unit_id = int(rcv_unit_id)
        units_ids = set([u.unit_id for u in self.units])
        if snd_unit_id not in units_ids or rcv_unit_id not in units_ids:
            raise self.UnitNotFoundError()
        if strength > 1 or strength < -1:
            raise self.WeightOutOfRangeError()
        self.units_conn.append((snd_unit_id, rcv_unit_id, strength))
        pass

    def connect_units_list(self, list_of_connections):
        """Connects all units in the list of triplets (sending_id,
        receiving_id, strength"""
        for c in list_of_connections:
            self.connect_units(*c)

    def connect_maps(self, snd_map, rcv_map):
        """indicates a general pattern of connectivity from one map to
        the other"""
        self.maps_conn.append((snd_map, rcv_map))

#######################
# VisualisableNetwork #
#######################

class VisualisableNetwork(object):
    """"Includes connections weights and units activity
    information."""
    def __init__(self, net_struct):
        "Initialize with the given VisualisableNetworkStructure."
        self.network_structure = net_struct
        # vtk_units[x] contains a tuple (pts, vtk_id) where pts is a
        # vtkPoints or a vtkPoints2D, vtk_id is a long that lets us do
        # a = [0, 0]
        # pts.GetPoint(vtk_id, a)
        self.vtk_units = [None] * len(net_struct.units)
    
    def represent_map(self, map_name):
        """Creates the vtkPoints collection of points representing the
        network map given in parameter (map name). Registers the points
        in the right position in the ordered vtk_units list."""
        net_struct_u = self.network_structure.units
        u_gids = self.network_structure.maps[map_name]
        pts = None
        if len(net_struct_u[net_struct_u.index(u_gids[0])].coords) == 2:
            pts = vtk.vtkPoints2D()
        else:
            pts = vtk.vtkPoints()
        grid = vtk.vtkUnstructuredGrid()     
        for p_gid in u_gids:
            p_i = net_struct_u.index(p_gid)
            coords = net_struct_u[p_i].coords
            print coords
            vtk_id = pts.InsertNextPoint(coords)
            self.vtk_units[p_i] = (grid, vtk_id)
        # vtkPoints (-> vtkPolyVertex?) -> vtkUnstructuredGrid
        grid.SetPoints(pts)
        return grid
            

###########################
# General setup functions #
###########################

# So visualization and processing are two different processes.  The
# simulation process may yield() a state display update every n
# epochs, to be piped into the visualisation process. Messaging may
# work the other way to control the simulation and the diplay
# (e.g. update frequency)

class vtkTimerCallback(object):
    def __init__(self, input_pipe):
        self.timer_count = 0
        self.child_conn = input_pipe
        self.network = None
 
    def execute(self,obj,event):
        global LOGGER, NETWORK, REN, REN_WIN, I_REN
        log_tick("vtkTimerCallback exec " + str(self.timer_count))
        # Non-blocking periodic reading of the pipe TODO: add code
        # that dynamically adjusts the periodicity of checking to a
        # fraction of the expected period of visualisable data input.
        # learning rate 0.1, not more because will swing around stable
        # point.  Bounded, 1ms to 2sec?
        if (not self.child_conn.poll()):
            log_tick("The pipeline is empty, returning")
            return
        r = interpret_visualisation_message(self.child_conn.recv())
        if r:
            NETWORK = r
            LOGGER.info("Network structure received.")
            m, a = map_source_object(NETWORK)
            add_actors_to_scene(REN, a)
            prepare_render_env(REN_WIN, I_REN)
            I_REN.Start()
        else:
            I_REN.GetRenderWindow().Render()
        self.timer_count += 1


def visualisation_process_f(child_conn, logger):
    """Function called when main() creates the visualisation process
    through multiprocessing.Process. The parameters are the pipe frrom
    which to read visualisation updates and the logger to use."""
    global LOGGER, REN, REN_WIN, I_REN
    LOGGER = logger
    log_tick("start visu")
    REN, REN_WIN, I_REN = setup_visualisation()
    REN.SetBackground(0.5, 0.5, 0.5)
    timer_id = setup_timer(I_REN, child_conn, d)
    


# set up a vtk pipeline
def setup_visualisation():
    ren = vtk.vtkRenderer()
    ren_win = vtk.vtkRenderWindow()
    ren_win.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(ren_win)
    return ren, ren_win, iren

    
def make_disk(outer, inner):
    source = vtk.vtkDiskSource()
    source.SetInnerRadius(1)
    source.SetOuterRadius(1.1)
    source.SetRadialResolution(20)
    source.SetCircumferentialResolution(20)
    source.Update()
    return source

def map_source_object(obj):
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInput(obj.GetOutput())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    return mapper, actor

def add_actors_to_scene(renderer, actor, *args):
    renderer.AddActor(actor)
    if args:
        map(renderer.AddActor, args)

def prepare_render_env(render_window, window_interactor):
    window_interactor.Initialize()
    render_window.Render()
    log_tick("after render")

def setup_timer(window_interactor, input_conn):
    callback = vtkTimerCallback(input_conn)
    window_interactor.AddObserver("TimerEvent", callback.execute)
    return window_interactor.CreateRepeatingTimer(100)

####################
# IPC and protocol #
####################

class ControlMessage(dict):
    def __init__(self, **kwds):
        self.update(kwds)
    def __getattr__(self, attr):
        if self.has_key(attr):
            return self[attr]
        else:
            return None

# def make_pickler(pipe_out):
#     return pickle.Pickler(pipe_out, pickle.HIGHEST_PROTOCOL)

# def make_unpickler(pipe_in):
#     return pickle.Unpickler(pipe_in)

# Message handling on the visualizer side

def interpret_visualisation_message(received_object):
    """Used by the visualisation process to trigger the action
    associated with a message received from the simulation process."""
    if isinstance(received_object, ControlMessage):
        handle_visualisation_control(received_object)

def handle_visualisation_control(obj):
    if (obj["exit"]):
        exit()

# Message handling on the main() simulation side

def interpret_control_message(received_object):
    pass

if __name__ == "__main__":
    # Insert unit tests here
    pass
