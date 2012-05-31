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

import itertools
import math
import operator
import pyNN.brian as pynnn
from visualisation import VisualisableNetworkStructure, Unit, ActivityUpdateMessage

class AdapterLockedError(Exception):
    pass

class PynnToVisuAdapter(object):
    def __init__(self, logger):
        self.commited = False
        self.logger = logger
        self.output_struct = VisualisableNetworkStructure()
        self.pynn_units_it = []
        # individual units list
        self.vis_units = []
        # ordered list of PyNN maps
        self.maps = []
        # dict of (pyNN population, previous (latest) list of
        # population units' spikes counts):
        self.spikes = dict()
        # individual connections list
        self.units_connections = []
        self.maps_connections = []
        self.num_units = 0
        # Aliases for names of populations {PyNN_name: alias, ...}
        self.aliases = {}

    def check_open(self):
        """Returns if the network structure is open for changes"""
        return not self.commited

    def assert_open(self):
        """Checks if the network structure is still open for changes,
        raising an exceptio if not"""
        if not self.check_open():
            raise AdapterLockedError()
            
    def commit_structure(self):
        """Adds all units to the output network structure, sorted by
        PyNN ID, and closes the adapter to prevent further structural
        change."""
        self.assert_open()
        self.vis_units = sorted(
            self.vis_units, key = lambda v_u_t: v_u_t[1].real)
        for u in self.vis_units:
            self.output_struct.add_unit(u[0], u[2])
        self.output_struct.connect_units_list(self.units_connections)
        for c in self.maps_connections:
            self.output_struct.connect_maps(c[0], c[1])
        self.commited = True

    def reopen(self):
        """TODO: to be able to reopen, we should keep track of units
        already commited to the output struct."""
        raise NotImplementedError

    def add_pynn_population(self, p, alias = None):
        """ Adds all units of the pyNN population p to the list of
        visualizable units"""
        self.assert_open()
        if alias == None:
            alias = p.label
        self.pynn_units_it = itertools.chain(self.pynn_units_it, p)
        three_d = "dz" in p.structure.parameter_names
        for pynn_u in p:
            u = None
            if not three_d:
                u = Unit(
                    int(pynn_u),
                    pynn_u.position[0], pynn_u.position[1])
            else:
                u = Unit(
                    int(pynn_u),
                    pynn_u.position[0], pynn_u.position[1],
                    pynn_u.position[2])
            self.vis_units.append((u, pynn_u, alias))
            # We add the unit when commiting the structure, because we
            # need to add them to the output_struct in global id order
        self.num_units += p.size
        self.maps.append(p)
        self.spikes[p] = [0]*p.size
        self.aliases[p.label] = alias

    def add_pynn_projection(self, sending_population,
                            receiving_population,
                            connection_manager):
        """Records a projection from sending_population to
        receiving_population. The connection_manager parameter
        expects a pyNN.brian.simulator.ConnectionManager, it can be
        obtained by
        pyNN.brian.simulator.Projection.connection_manager.
        """
        self.assert_open()
        w = connection_manager.get("weight", "array")
        n_send, n_recv = w.shape
        for i in range(0, n_send):
            pynn_u = sending_population[i]
            # list of tuples containing the sending unit's global
            # index (same for each tuple), the global indices of the
            # receiving cells in existing connections from neuron i,
            # and the weights
            connections = [
                (pynn_u.real, 
                 receiving_population[a].real, 
                 self.convert_weight(b))
                for a, b in enumerate(w[i]) if not math.isnan(b)]
            self.units_connections.extend(connections)
        self.maps_connections.append((sending_population.label,
                                        receiving_population.label))

    def convert_weight(self, w):
        """Converts pyNN weights by normalizing them from their
        natural range to [-1, 1]."""
        # TODO: this is a dummy method. it should prompt pyNN for the
        # range and use a sigmoidal adjustment.
        if w > 1 or w < -1:
            w = min(max(w, -1), 1)
            self.logger.info("convert_weights made a dummy adjustment.")
        return w

    def make_activity_update_message(self):
        newly_fired = dict()
        max_current_activity_list = list()
        for m in self.maps:
            spikes = m.get_spike_counts().values()
            newly_fired[m] = map(operator.sub, spikes, self.spikes[m])
            self.spikes[m] = spikes
            max_current_activity_list.append(max(newly_fired[m]))
            self.logger.debug(("A map with %s units and a max activity of "
                               "%s passes by."), len(newly_fired[m]),
                              max(newly_fired[m]))
        max_current_activity = max(max_current_activity_list)
        activity = list()
        if max_current_activity == 0:
            activity = [0] * self.num_units
            self.logger.debug(("There is no overall activity, making an"
                               "update list of %s zeros."), self.num_units)
        else:
            for m in self.maps:
                activity = activity + \
                    map(lambda x: min(1., 
                                      x / float(max_current_activity)),
                        newly_fired[m])
        self.logger.debug(("The update list contains %s values. It "
                           "looks like that: %s"), len(activity),
                          activity)
        return ActivityUpdateMessage(activity)
