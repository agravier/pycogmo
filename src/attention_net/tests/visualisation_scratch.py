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


