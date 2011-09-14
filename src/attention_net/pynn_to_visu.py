#!/usr/bin/env python2.7

import itertools
import math
import pyNN.nest as pynnn
from visualisation import VisualisableNetworkStructure

class NetworkClosedForStructuralChanges(Exception):
    pass

class PynnToVisuAdapter(object):
    def __init__(self, logger):
        self.commited = False
        self.logger = logger
        self.output_struct = VisualisableNetworkStructure()
        self.pynn_units_it = []
        self.vis_units = []
        self.num_units = 0

    def check_open(self):
        """Checks if the network structure is still open for changes,
        raising an exceptio if not"""
        if self.commited:
            raise NetworkClosedForStructuralChanges()

    def add_pynn_population(self, p, concept_map = None):
        self.check_open()
        self.pynn_units_it = itertools.chain(self.pynn_units_it, p)
        three_d = "dz" in p.structure.parameter_names
        for pynn_u in p:
            u = None
            if three_d:
                u = VisualisableNetworkStructure.Unit(
                    pynn_u.real,
                    pynn_u.position[0], pynn_u.position[1])
            else:
                u = VisualisableNetworkStructure.Unit(
                    pynn_u.real,
                    pynn_u.position[0], pynn_u.position[1],
                    pynn_u.position[2])
            self.vis_units.append((u, pynn_u, concept_map))
            # We add the unit when commiting the structure, because we
            # need to add them to the output_struct in global id order 
        self.num_units += p.size
            
    def commit_structure(self):
        self.check_open()
        self.vis_units = sorted(
            self.vis_units, key = lambda v_u_t: v_u_t[1].real)
        for u in self.vis_units:
            self.output_struct.add_unit(u[0], concept_map)
        self.commited = True

    def add_pynn_projection(self, sending_population,
                            receiving_population,
                            connection_manager):
        self.check_open()
        w = connection_manager.get("weight", "array")
        n_send, n_recv = w.shape
        for i in range(0, n_send-1):
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
            self.output_struct.connect_units(connections)
        self.output_struct.connect_maps(sending_population.label,
                                        receiving_population.label)

    def convert_weight(self, w):
        """ Converts pyNN weights by normalizing them from their
        natural range to [-1, 1]."""
        # TODO: this is a dummy method. it should prompt pyNN for the
        # range and use a sigmoidal adjustment.
        if w > 1 or w < -1:
            w = min(max(w, -1), 1)
            logger.info("convert_weights made a dummy adjustment.")
        return w
