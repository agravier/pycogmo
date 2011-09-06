#!/usr/bin/env python2.7

import pyNN.nest as pynnn
#from pyNN.recording.files import HDF5ArrayFile needs cython and tables (=pain)
from pyNN.utility import init_logging
import datetime
import logging
import multiprocessing # because threading will not bypass the GIL
import os
import sys
import time
import vtk

from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL, FATAL
from multiprocessing import SUBDEBUG, SUBWARNING
LOGGER = multiprocessing.get_logger()
LOG_DIR = "./logs"

def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

def configure_loggers():
   LOGGER.setLevel(SUBDEBUG)
   debug_handler = logging.StreamHandler()
   debug_handler.setLevel(SUBDEBUG)
   debug_formatter = logging.Formatter(
      fmt='%(asctime)s - %(levelname)s - %(message)s',
      datefmt='%m-%d %H:%M')
   debug_handler.setFormatter(debug_formatter)
   LOGGER.addHandler(debug_handler)
   logfile = LOG_DIR + "/" + datetime.datetime.now().strftime("%Y.%m.%d-%H.%M.%S.%f") + ".log"
   ensure_dir(logfile)
   file_handler = logging.FileHandler(logfile)
   file_handler.setLevel(INFO)
   file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
   file_handler.setFormatter(file_formatter)
   LOGGER.addHandler(file_handler)

# So visualization and processing are two different processes.
# The simulation process may yield() a state display update every n epochs,
# to be piped into the visualization process. Messaging may work the other way
# to control the simulation and the diplay (e.g. update frequency)

class vtkTimerCallback(object):
   def __init__(self, input_pipe, disk):
       self.timer_count = 0
       self.child_conn = input_pipe
       self.disk = disk
 
   def execute(self,obj,event):
       log_tick("vtkTimerCallback exec " + str(self.timer_count))
       # Non-blocking pipe reading with periodic checking is the solution
       # TODO: add code that dynamically adjusts periodicity of checking to
       # afraction of the expected period of visualisable data input.
       # learning rate 0.1, not more because will swing around stable point.
       # Bounded, 1ms to 2sec?
       if (not self.child_conn.poll()):
           log_tick("the pipeline is empty, returning")
           return
       r = self.child_conn.recv()
       if r<=0:
           exit()
       log_tick("tick received")
       self.disk.SetInnerRadius(1-r/100.)
       #actor.SetPosition(self.timer_count, self.timer_count,0);
       iren = obj
       iren.GetRenderWindow().Render()
       self.timer_count += 1

def log_tick(s):
    now = datetime.datetime.now()
    LOGGER.log(SUBDEBUG, "tick at time %s: %s", now, s)
    for h in LOGGER.handlers:
       h.flush()


def visualization_process_f(child_conn):
    log_tick("start visu")
    REN, REN_WIN, I_REN = setup_visualization()
    REN.SetBackground(0.5, 0.5, 0.5)
    d = make_disk(2,1)
    m, a = map_source_object(d)
    add_actors_to_scene(REN, a)
    prepare_render_env(REN_WIN, I_REN)
    timer_id = setup_timer(I_REN, child_conn, d)
    I_REN.Start()



# set up a vtk pipeline
def setup_visualization():
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

def setup_timer(window_interactor, input_conn, disk_to_update):
    callback = vtkTimerCallback(input_conn, disk_to_update)
    window_interactor.AddObserver("TimerEvent", callback.execute)
    return window_interactor.CreateRepeatingTimer(100)

# Consiser using protocol buffers to picke thatif python's pickling
# causes problems
class VisualizableNetworkStructure(object):
    """Initial message sent to a visualizer child process to set up
    the physical elements in 3d space. This structure is abstract of
    particular visual choices, projections, transformations, or
    whether a neural map should be represented as a sheet, a cloud of
    units, or anything else. It's the visualization interface that is
    to make or let the user make these choices."""
    def __init__(self):
        # all units in consistent order, the same order that is to be
        # used when transmitting activity updates.
        self.units = list()
        # info abou the conceptual grouping of units
        self.maps = list()
        # detailed connectivity (unit-to-unit)
        self.units_conn = list()
        # abstract connectivity (between maps)
        self.maps_conn = list()

    # setting the units collection
    def add_unit(pynn_unit, assign_map = None):
       pass
    
    def add_population(pynn_population, override_map = None):
       pass
    
    def connect_units(u1, u2, strength):
       pass
       
def main():
   configure_loggers()
   parent_conn, child_conn = multiprocessing.Pipe()
   p = multiprocessing.Process(target=visualization_process_f,
                               name="display_process",
                               args=(child_conn,))
   p.start()

   # parent->child test
   for a in reversed(range(-1,100)):
       time.sleep(0.1)
       log_tick("just before send")
       # Only pipe in data to be visualized if visualization pipe is empty
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
   prj1_2 = pynnn.Projection(p1, p2,
                       pynnn.AllToAllConnector(allow_self_connections=False),
                       target='excitatory')
   # I may need to make own PyNN Connector class. Otherwise, this is neat:
   # exponentially decaying probability of connections depends on distance.
   # distance is only calculated using x and y, which are on a toroidal topo
   # with boundaries at 0 and 500
   connector = pynnn.DistanceDependentProbabilityConnector("exp(-abs(d))",
                   space=pynnn.Space(axes='xy',
                              periodic_boundaries=((0,500), (0,500), None)))
   # Otherwise, the very leet connection set algebra (python CSA module) can
   # be used.
   weight_distr = pynnn.RandomDistribution(distribution='gamma',
                                           parameters=[1,0.1])
   prj1_2.randomizeWeights(weight_distr)
   pynnn.run(100.0)
   pynnn.end()


if __name__ == "__main__":
   main()
