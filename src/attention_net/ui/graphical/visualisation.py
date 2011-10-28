#!/usr/bin/env python2

import cPickle as picke
import datetime
import itertools
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
            LOGGER.error(("Units with ID %i and %i and "
                         "coordinates %s and %s can't co-exist."),
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
    


# Consiser using protocol buffers to pickle that if python's pickling
# causes problems
class VisualisableNetworkStructure(object):
    """Initial message sent to a visualiser child process to set up
    the physical elements in 3d space. This structure is abstract of
    particular visual choices, projections, transformations, or
    whether a neural map should be represented as a sheet, a cloud of
    units, or anything else. It's the visualisation interface that is
    to make or let the user make these choices."""
    
    def __init__(self):
        # all units (class Unit) in
        # consistent order, the same order that is to be used when
        # receiving activity updates.
        self.units = list()
        # all maps' unique IDs in the same order that is to be
        # used when receiving activity updates.
        self.maps_ids = list()
        # info about the conceptual grouping of units, a dict of
        # (UNIQUE ID of the map, list of unit global indices in
        # activity update reception order)
        self.maps = dict()
        # detailed connectivity (unit-to-unit)
        self.units_conn = list()
        # abstract connectivity (between maps UNIQUE IDs)
        self.maps_conn = list()
        # aliases for maps: dict(map ID, alias)
        self.maps_aliases = dict()

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
        Unit to the given map that may
        need to be created for the occasion."""
        if unit not in self.units:
            raise self.UnitNotFoundError()
        if assign_map not in self.maps:
            self.maps[assign_map]=[]
        if assign_map not in self.maps_ids:
            self.maps_ids.append(assign_map)
        self.maps[assign_map].append(int(unit))
    
    def add_population(self, iterable_population, override_map_id = None):
        """appends a group of units to the list of units."""
        for u in iterable_population:
            self.add_unit(u, override_map_id)
    
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

    def map_alias(self, m, alias):
        "Register alias for map m"
        self.maps_aliases[m] = alias

#######################
# VisualisableNetwork #
#######################

class VisualisableNetwork(object):
    """"Includes connections weights and units activity
    information."""
    def __init__(self, net_struct):
        "Initialize with the given VisualisableNetworkStructure."
        self.network_structure = net_struct
        self.vtk_units = [None] * len(net_struct.units)
        self.lut = vtk.vtkColorTransferFunction(); 
        self.lut.SetColorSpaceToHSV()
        self.lut.SetScaleToLinear()
        # Work with white and with black bg
        self.lut.AddHSVPoint(0., 1., 0., 0.5); 
        self.lut.AddHSVPoint(1., 1., 1., 1.); 
        self.lut.SetRange(0.,1.)
        # Build update-ordered representations of network maps
        self.grids = list()
        self.grids_lengths = list()
        for g, l in [self.represent_map(m) for m in
                     self.network_structure.maps_ids]:
            self.grids.append(g)
            self.grids_lengths.append(l)
        # vtk_units[x] contains a tuple (pts, vtk_id) where pts is a
        # vtkPoints or a vtkPoints2D, vtk_id is a long that lets us do
        # a = [0, 0]; pts.GetPoint(vtk_id, a)
        # maps_actors is a dict(map id, actor representing the map)
    
    def represent_map(self, map_id):
        """Creates the vtkPoints collection of points representing the
        network map given in parameter (map name), and the wrapping
        vtkPolyVertex and vtkUnstructuredGrid. Registers the points in
        the right position in the ordered vtk_units list."""
        net_struct_u = self.network_structure.units
        u_gids = self.network_structure.maps[map_id]
        pts = None
        # if len(net_struct_u[net_struct_u.index(u_gids[0])].coords) == 2:
        #     pts = vtk.vtkPoints2D()
        # else:
        #     pts = vtk.vtkPoints()
        pts = vtk.vtkPoints()
        grid = vtk.vtkUnstructuredGrid()
        pv = vtk.vtkPolyVertex()
        l = []
        for p_gid in u_gids:
            p_i = net_struct_u.index(p_gid)
            coords = net_struct_u[p_i].coords
            if len(coords) == 2:
                coords = (coords[0], coords[1], 0)
            l.append(pts.InsertNextPoint(coords))
            self.vtk_units[p_i] = (grid, l[len(l)-1])
        # necessay if using SetId and not InsertId in the next loop:
        pv.GetPointIds().SetNumberOfIds(len(l)) 
        for i in range(len(l)):
            pv.GetPointIds().SetId(i, l[i])
        # vtkPoints -> vtkPolyVertex -> vtkUnstructuredGrid
        grid.InsertNextCell(pv.GetCellType(), pv.GetPointIds())
        grid.SetPoints(pts)
        return (grid, len(l))

    def make_actor_for_grid(self, grid, translation=[0.,0.,0.]):
        """returns an actor with an attached mapper for the
        unstructured grid. scalar data on units is represented by
        color, translated by the given vector.""" 
        mapper = vtk.vtkDataSetMapper()
        mapper.SetInput(grid)
        mapper.SetColorModeToMapScalars()
        mapper.SetLookupTable(self.lut); 
        ## Calls SelectColorArray with a [0,1] range mapping
        # mapper.SelectColorArray(self.DEFAULT_COLOR_ARRAY)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.AddPosition(translation[0], translation[1], translation[2])
        return actor

    def make_all_actors(self, levels_to_grids):
        """Given a dict mapping from desired display level to list
        of vtkUnstructuredGrid, returns a list of VTK actors, each
        translated to the desired level, avoiding collision with other
        grids."""
        def grid_max_dim(vtk_u_g):
            b = vtk_u_g.GetBounds()
            return max(b[1]-b[0], b[3]-b[2], b[5]-b[4])
        list_of_grids = list(
            itertools.chain.from_iterable(levels_to_grids.values())) 
        # Min distance between two grids to avoid collision:
        largest_grid_dim = max(map(grid_max_dim, list_of_grids))
        if largest_grid_dim == 0:
            largest_grid_dim = 1
        # Create actors, determine translations
        translated_actors = list() # list of translated actors
        for lvl in levels_to_grids.keys():
            # Vertical center of translated grids for this level:
            z_trans = lvl*largest_grid_dim + largest_grid_dim*lvl*0.1
            current_lvl_grids = levels_to_grids[lvl]
            g_trans_list = list()
            len_cur_lvl_grids = len(current_lvl_grids)
            for i in range(len_cur_lvl_grids):
                g = current_lvl_grids[i]
                g_b = g.GetBounds()
                g_height = g_b[5]-g_b[4]
                g_width = g_b[1] - g_b[0]
                if g_width == 0:
                    g_width = 1
                # Horizontal translation on x with the left corner of
                # the first grid at x=0 (centered on x=0 after the
                # loop):
                g_x_trans = g_width*i+g_width*i*0.2-g_b[0]
                # Horixontal alignment of the front of the bounding
                # box of each map along the xz plane (y=0):
                g_y_trans = -g_b[2]
                # Vertical adjustment of each grid:
                g_z_trans = z_trans - g_b[4] - g_height/2.
                g_trans_list.append((g_x_trans, g_y_trans, g_z_trans))
            # Center x
            last_bounds = current_lvl_grids[len_cur_lvl_grids-1].GetBounds()
            x_max = g_trans_list[len_cur_lvl_grids-1][0] + \
                last_bounds[1] - last_bounds[0]
            for i in range(len_cur_lvl_grids):
                translated_actors.append(
                    self.make_actor_for_grid(
                        current_lvl_grids[i], 
                        translation=(g_trans_list[i][0] - x_max/2.,
                                     g_trans_list[i][1],
                                     g_trans_list[i][2])))
        return translated_actors
    
    def update_scalars(self, 
                       scalars_list, 
                       grids=None, 
                       lengths=None):
        """grids and lengths are two complementary lists, g is a list
        of vtkUnstructuredGrid, lengths is the list of the number n of
        units in each g, and scalars_list is the list of updated values
        for all units in all grids g. scalars_list is ordered following the
        grid_lengths list of grids. The method updates the scalar
        vaues of all units in all grids using scalars_list."""
        if grids == None:
            grids=self.grids
        if lengths == None:
            lengths=self.grids_lengths
        m = 0
        for i in range(len(grids)): 
            l = lengths[i]
            scalars_slice = scalars_list[m:l]
            m = l
            vtk_scalars = vtk.vtkFloatArray()
            for j in range(l):
                vtk_scalars.InsertNextValue(scalars_slice[j])
            grids[i].GetPointData().SetScalars(vtk_scalars)

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
        r = interpret_simu_to_visu_message(self.child_conn.recv())
        if r:
            NETWORK = r
            LOGGER.info("Network structure received.")
            lvl_to_g = {0 : NETWORK.grids}
            # lvl_to_g = Tns.vn.levels_to_grids()
            all_actors = NETWORK.make_all_actors(lvl_to_g)
            add_actors_to_scene(REN, all_actors)
            REN.ResetCamera()
            prepare_render_env(REN_WIN, I_REN)
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
    log_tick("after setup_vis")
    REN.SetBackground(0.5, 0.5, 0.5)
    log_tick("background set")
    timer_id = setup_timer(I_REN, child_conn)
    I_REN.Start()

# set up a vtk pipeline
def setup_visualisation():
    ren = vtk.vtkRenderer()
    ren_win = vtk.vtkRenderWindow()
    ren_win.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(ren_win)
    iren.Initialize()
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

def add_actors_to_scene(renderer, actors):
    map(renderer.AddActor, actors)

def prepare_render_env(render_window, window_interactor):
    window_interactor.Initialize()
    render_window.Render()
    log_tick("after render")

def setup_timer(window_interactor, input_conn):
    callback = vtkTimerCallback(input_conn)
    window_interactor.AddObserver("TimerEvent", callback.execute)
    log_tick("observer added")
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

class ActivityUpdateMessage(object):
    def __init__(self, units_activities):
        self.units_activities = units_activities
    @property
    def units_activities(self):
        return units_activities

# def make_pickler(pipe_out):
#     return pickle.Pickler(pipe_out, pickle.HIGHEST_PROTOCOL)

# def make_unpickler(pipe_in):
#     return pickle.Unpickler(pipe_in)

# Message handling on the visualizer side

def interpret_simu_to_visu_message(received_object):
    """Used by the visualisation process to trigger the action
    associated with a message received from the simulation process."""
    if isinstance(received_object, ControlMessage):
        handle_visualisation_control(received_object)
    if isinstance(received_object, ActivityUpdateMessage):
        NETWORK.update_scalars(received_object.units_activities)
    if isinstance(received_object, VisualisableNetworkStructure):
        return VisualisableNetwork(received_object)

def handle_visualisation_control(obj):
    if (obj["exit"]):
        LOGGER.info("Visualisation received exit control msg.")
        exit()

# Message handling on the main() simulation side

def interpret_visu_to_simu_message(received_object):
    pass

def handle_simulation_control(obj):
    pass

if __name__ == "__main__":
    # Insert unit tests here
    pass
