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

""" This is a sandbox file used to try out VTK.
"""

from vtk import *

pts = vtk.vtkPoints()

grid = vtk.vtkUnstructuredGrid()

pv = vtkPolyVertex()

class Tns(object):
    pass

import itertools

Tns.l_id, Tns.l_x, Tns.l_y, Tns.l_z = xrange(5, 20), xrange(3, 18), xrange(0,30,2), itertools.repeat(-1,15)

Tns.units_coords = [(x, y, z) for (u_id, x, y, z) in itertools.izip(Tns.l_id, Tns.l_x, Tns.l_y, Tns.l_z)]

l = []

for c in Tns.units_coords:
    l.append(pts.InsertNextPoint(c))

pv.GetPointIds().SetNumberOfIds(len(l)) # necessay if using SetId and not InsertId in the next loop 
for i in range(len(l)):
    pv.GetPointIds().SetId(i, l[i])

grid.InsertNextCell(pv.GetCellType(), pv.GetPointIds())
grid.SetPoints(pts)

aPolyVertexMapper = vtk.vtkDataSetMapper()
aPolyVertexMapper.SetInput(grid)
aPolyVertexActor = vtk.vtkActor()
aPolyVertexActor.SetMapper(aPolyVertexMapper)
# Create the usual rendering stuff.
ren = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)
renWin.SetSize(300, 150)
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)
ren.SetBackground(.1, .2, .4)
ren.AddActor(aPolyVertexActor)
# Render the scene and start interaction.
iren.Initialize()
renWin.Render()
iren.Start()


from ui.graphical.visualisation import VisualisableNetwork as VN

from ui.graphical.visualisation import VisualisableNetworkStructure as VNS

Tns.vns_units = [VNS.Unit(u_id, x, y, z) for (u_id, x, y, z) in itertools.izip(Tns.l_id, Tns.l_x, Tns.l_y, Tns.l_z)]


