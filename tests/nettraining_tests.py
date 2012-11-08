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
import logging
from logging import NullHandler 
from mock import Mock, patch
from nose import with_setup
from nose.tools import eq_, raises, timed, nottest
from numpy import array as npa
from scheduling.nettraining import *
# numpy.testing.utils.assert_almost_equal is more useful for arrays than the
# nose.tools version:
from numpy.testing import assert_allclose
from numpy.testing.utils import assert_array_less
import pyNN.brian as pynnn
from tests.pynn_utils_tests import setup_pynn_populations, \
    setup_registered_rectinilinear_ouput_rate_encoders, Tns
from common.pynn_utils import enable_recording, InputSample, \
    RectilinearOutputRateEncoder, get_rate_encoder
from tests.pynn_scheduling_tests import setup_clean_simpy
from scheduling.pynn_scheduling import get_current_time, configure_scheduling
import scheduling
import common.pynn_utils

DUMMY_LOGGER = logging.getLogger("testLogger")
DUMMY_LOGGER.addHandler(NullHandler())

# TODO: Check for runaway positive feedback of SOM units


def setup_data():
    Tns.s_in_1 = 0.8
    Tns.s_in_2 = 77.7
    Tns.s_in_3 = -0.5
    Tns.s_w_1 = 0
    Tns.s_w_2 = 0.5
    Tns.s_w_3 = 1.1
    Tns.v_in_1 = [0, 1, 0, 0.2, 0.5]
    Tns.v_in_2 = [13, 0, 4.4, -1, -0.5]
    Tns.v_in_3 = [1, 1, 1, 1, 1]
    Tns.v_w_1 = [0, 0.1, 0.3, 0.7, 1]
    Tns.v_w_2 = [-13, 0.4, -1, 2, 5]
    Tns.v_w_3 = [1, 1, 1, 1, 1]
    Tns.s_out_1 = 0.9
    Tns.s_out_2 = 0
    Tns.s_out_3 = -1


def setup_input_samples():
    Tns.sample1 = InputSample(8, 8, ([[1, 0] * 4] + [[0, 1] * 4]) * 4)#[[1] * 8] * 4 + [[0] * 8] * 4)
    Tns.sample2 = InputSample(8, 8, ([[0, 1] * 4] + [[1, 0] * 4]) * 4)


def setup_4_units_input_samples():
    Tns.sample1 = InputSample(2, 2, [[1, 0], [0, 1]])
    Tns.sample2 = InputSample(2, 2, [[0, 1], [1, 0]])


def setup_2_layers_ff_net():
    configure_scheduling()
    setup_registered_rectinilinear_ouput_rate_encoders()
    enable_recording(Tns.p1, Tns.p2)
    schedule_output_rate_calculation(Tns.p1)
    schedule_output_rate_calculation(Tns.p2)
    

def setup_2_layers_4_units_ff_net():
    configure_scheduling()
    pynnn.setup()
    Tns.p1 = pynnn.Population(4, pynnn.IF_curr_alpha,
                          structure=pynnn.space.Grid2D())
    Tns.p2 = pynnn.Population(4, pynnn.IF_curr_alpha,
                          structure=pynnn.space.Grid2D())
    Tns.prj1_2 = pynnn.Projection(
        Tns.p1, Tns.p2, pynnn.AllToAllConnector(allow_self_connections=False),
        target='excitatory')
    Tns.prj1_2.set("weight", 1)
    Tns.max_weight = 34
    Tns.rore1_update_p = 10
    Tns.rore1_win_width = 200
    Tns.rore2_update_p = 10
    Tns.rore2_win_width = 200
    Tns.rore1 = RectilinearOutputRateEncoder(Tns.p1, 2, 2,
                                             Tns.rore1_update_p,
                                             Tns.rore1_win_width)
    Tns.rore2 = RectilinearOutputRateEncoder(Tns.p2, 2, 2,
                                             Tns.rore2_update_p,
                                             Tns.rore2_win_width)
    common.pynn_utils.POP_ADAPT_DICT[(Tns.p1,
        common.pynn_utils.RectilinearOutputRateEncoder)] = Tns.rore1
    common.pynn_utils.POP_ADAPT_DICT[(Tns.p2,
        common.pynn_utils.RectilinearOutputRateEncoder)] = Tns.rore2
    enable_recording(Tns.p1, Tns.p2)
    schedule_output_rate_calculation(Tns.p1)
    schedule_output_rate_calculation(Tns.p2)


@with_setup(setup_data)
def test_hebb_learning_scalar():
    assert hebb_learning(Tns.s_in_1, Tns.s_out_1, Tns.s_w_1, 1) == 0.9 * 0.8
    assert hebb_learning(Tns.s_in_2, Tns.s_out_2, Tns.s_w_2, 0.5) == Tns.s_w_2
    assert hebb_learning(Tns.s_in_3, Tns.s_out_3, Tns.s_w_3, 3.5) == \
        3.5 * Tns.s_out_3 * Tns.s_in_3 + Tns.s_w_3


@with_setup(setup_data)
def test_hebb_learning_vector():
    assert len(hebb_learning(Tns.v_in_1, Tns.s_out_1, Tns.v_w_1, 1)) == 5
    assert (hebb_learning(Tns.v_in_1, Tns.s_out_1, Tns.v_w_1, 1) == \
        Tns.v_w_1 + npa(Tns.s_out_1) * npa(Tns.v_in_1)).all()
    assert len(hebb_learning(Tns.v_in_2, Tns.s_out_2, Tns.v_w_2, 0.5)) == 5
    assert (hebb_learning(Tns.v_in_2, Tns.s_out_2, Tns.v_w_2, 0.5) == \
        Tns.v_w_2 + .5 * npa(Tns.s_out_2) * npa(Tns.v_in_2)).all()
    assert len(hebb_learning(Tns.v_in_3, Tns.s_out_3, Tns.v_w_3, 3.5)) == 5
    assert (hebb_learning(Tns.v_in_3, Tns.s_out_3, Tns.v_w_3, 3.5) == \
        Tns.v_w_3 + 3.5 * npa(Tns.s_out_3) * npa(Tns.v_in_3)).all()


@with_setup(setup_data)
def test_oja_learning_scalar():
    assert_allclose(oja_learning(Tns.s_in_1, Tns.s_out_1, Tns.s_w_1, 1),
                        0.72)
    assert oja_learning(Tns.s_in_2, Tns.s_out_2, Tns.s_w_2, 0.5) == Tns.s_w_2
    assert_allclose(oja_learning(Tns.s_in_3, Tns.s_out_3, Tns.s_w_3, 3.5),
                        -1)


@with_setup(setup_data)
def test_oja_learning_vector():
    assert len(oja_learning(Tns.v_in_1, Tns.s_out_1, Tns.v_w_1, 1)) == 5
    assert_allclose(oja_learning(Tns.v_in_1, Tns.s_out_1, Tns.v_w_1, 1),
        [0., 0.919, 0.057, 0.313, 0.64])
    assert len(oja_learning(Tns.v_in_2, Tns.s_out_2, Tns.v_w_2, 0.5)) == 5
    assert (oja_learning(Tns.v_in_2, Tns.s_out_2, Tns.v_w_2, 0.5) == \
        [-13., 0.4, -1., 2., 5.]).all()
    assert len(oja_learning(Tns.v_in_3, Tns.s_out_3, Tns.v_w_3, 3.5)) == 5
    assert (oja_learning(Tns.v_in_3, Tns.s_out_3, Tns.v_w_3, 3.5) == \
        [-6] * 5).all()


@with_setup(setup_data)
def test_conditional_pca_learning_scalar():
    assert_allclose(conditional_pca_learning(Tns.s_in_1, Tns.s_out_1,
        Tns.s_w_1, 1), 0.72)
    assert conditional_pca_learning(Tns.s_in_2, Tns.s_out_2, Tns.s_w_2, 0.5) \
        == Tns.s_w_2
    assert_allclose(conditional_pca_learning(Tns.s_in_3, Tns.s_out_3,
        Tns.s_w_3, 3.5), 6.7)


@with_setup(setup_data)
def test_conditional_pca_learning_vector():
    assert len(conditional_pca_learning(Tns.v_in_1, Tns.s_out_1, \
        Tns.v_w_1, 1)) == 5
    assert_allclose(conditional_pca_learning(Tns.v_in_1, Tns.s_out_1, \
        Tns.v_w_1, 1), [0., 0.91, 0.03, 0.25, 0.55])
    assert len(conditional_pca_learning(Tns.v_in_2, Tns.s_out_2, \
        Tns.v_w_2, 0.5)) == 5
    assert (conditional_pca_learning(Tns.v_in_2, Tns.s_out_2, Tns.v_w_2, 0.5) \
        == [-13., 0.4, -1., 2., 5.]).all()
    assert len(conditional_pca_learning(Tns.v_in_3, Tns.s_out_3, \
        Tns.v_w_3, 3.5)) == 5
    assert (conditional_pca_learning(Tns.v_in_3, Tns.s_out_3, Tns.v_w_3, 3.5) \
            == [1.] * 5).all()


@with_setup(setup_2_layers_ff_net)
def test_kwta_presentation():
    """Tests one kwta presentation to half of the units, followed by
    another presentation to the second half."""
    s = InputSample(8, 8, [[1] * 8] * 4 + [[0] * 8] * 4)
    kwta_presentation(Tns.p2, Tns.p1, s, 2)
    assert get_current_time() == 2
    rates = get_rate_encoder(Tns.p1).get_rates()
    assert_array_less(rates[4:8], rates[0:4])
    s = InputSample(8, 8, [[0] * 8] * 4 + [[1] * 8] * 4)
    kwta_presentation(Tns.p2, Tns.p1, s, 2)
    assert get_current_time() == 4
    rates = get_rate_encoder(Tns.p1).get_rates()
    assert_array_less(rates[0:4], rates[4:8])

    
def test_select_kwta_winners():
    shape = (16, 4)
    mock_rate_encoder = Mock(spec_set=RectilinearOutputRateEncoder)
    mock_rates = numpy.array(
        [[10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
         [0.001, 0, 3.0, 21., 0.0001, 3, 5.0, 11, 18.18, 9, 10, 11, 12, 0.16, 15, 14],
         [1, 0.5, 0, 0., 0.0001, 21, 0, 1, 0.18, 5, 0, 1, 34, 4, 16, 14],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    mock_rate_encoder.get_rates.return_value = mock_rates
    mock_rate_encoder.shape = shape
    mock_pop = Mock()
    import common, scheduling
    try:
        scheduling.nettraining.get_rate_encoder = Mock()
        scheduling.nettraining.get_rate_encoder.return_value = mock_rate_encoder
        w = select_kwta_winners(mock_pop, k=4, presentation_duration=100)
        eq_(set(w), set([(2, 12), (1, 3), (1, 8), (2, 5)]))
    finally:
        scheduling.nettraining.get_rate_encoder = common.pynn_utils.get_rate_encoder


@with_setup(setup_input_samples)
@with_setup(setup_2_layers_ff_net)
def test_kwta_epoch_1_winner():
    Tns.p2.max_unit_rate=10
    Tns.p1.max_unit_rate=3.3
    sum_weights_before_1st_epoch = \
        numpy.add.reduce(get_weights(Tns.prj1_2, Tns.max_weight)._weights, axis=0)
    kwta_epoch(trained_population=Tns.p2,
               input_population=Tns.p1,
               projection=Tns.prj1_2,
               input_samples=itertools.repeat(Tns.sample1, 2),
               num_winners=1,
               neighbourhood_fn=None,
               presentation_duration=25,
               learning_rule=conditional_pca_learning,
               learning_rate=0.1,
               max_weight_value=Tns.max_weight)
    sum_weights_after_1st_epoch = \
        numpy.add.reduce(get_weights(Tns.prj1_2, Tns.max_weight)._weights, axis=0)
    weights_diff_1 = sum_weights_after_1st_epoch - sum_weights_before_1st_epoch
    argwinner_1 = numpy.argmax(weights_diff_1)
    # assert diff is all 0 except winner
    winner_diff_1 = weights_diff_1[argwinner_1]
    assert winner_diff_1 > 0
    weights_diff_1_nowin = numpy.array(weights_diff_1)
    weights_diff_1_nowin[argwinner_1] -= winner_diff_1
    assert_allclose(weights_diff_1_nowin, numpy.zeros(len(weights_diff_1)))
    # run some time without input to let the activity come back to 0
    from scheduling.pynn_scheduling import RATE_ENC_RESPAWN_DICT
    run_simulation(get_current_time()+30)
    # second epoch
    kwta_epoch(trained_population=Tns.p2,
               input_population=Tns.p1,
               projection=Tns.prj1_2,
               input_samples=itertools.repeat(Tns.sample2, 2),
               num_winners=1,
               neighbourhood_fn=None,
               presentation_duration=25,
               learning_rule=conditional_pca_learning,
               learning_rate=0.1,
               max_weight_value=Tns.max_weight)
    sum_weights_after_2nd_epoch = \
        numpy.add.reduce(get_weights(Tns.prj1_2, Tns.max_weight)._weights, axis=0)
    weights_diff_2 = sum_weights_after_2nd_epoch - sum_weights_after_1st_epoch
    argwinner_2 = numpy.argmax(weights_diff_2)
    assert argwinner_1 != argwinner_2
    # assert still only one non zero diff
    winner_diff_2 = weights_diff_2[argwinner_2]
    assert winner_diff_2 > 0
    weights_diff_2_nowin = numpy.array(weights_diff_2)
    weights_diff_2_nowin[argwinner_2] -= winner_diff_2
    assert_allclose(weights_diff_2_nowin, numpy.zeros(len(weights_diff_2)))


@with_setup(setup_4_units_input_samples)
@with_setup(setup_2_layers_4_units_ff_net)
def test_kwta_epoch_weights_below_1():
    sum_weights_after_epoch = numpy.ndarray([], dtype=numpy.float)
    import sys
    for i in xrange(8):
        kwta_epoch(trained_population=Tns.p2,
                   input_population=Tns.p1,
                   projection=Tns.prj1_2,
                   input_samples=itertools.repeat(Tns.sample1, 1),
                   num_winners=1,
                   neighbourhood_fn=None,
                   presentation_duration=22,
                   learning_rule=conditional_pca_learning,
                   learning_rate=1,
                   max_weight_value=Tns.max_weight,
                   trained_pop_max_rate=10,
                   input_pop_max_rate=3.4)
    assert (get_weights(Tns.prj1_2, Tns.max_weight).normalized_numpy_weights <= 1).all()
    

@with_setup(setup_input_samples)
@with_setup(setup_2_layers_ff_net)
def test_kwta_epoch_2_winners():
    Tns.p2.max_unit_rate=10
    Tns.p1.max_unit_rate=3.3
    sum_weights_before_1st_epoch = \
        numpy.add.reduce(get_weights(Tns.prj1_2, Tns.max_weight)._weights, axis=0)
    kwta_epoch(trained_population=Tns.p2,
               input_population=Tns.p1,
               projection=Tns.prj1_2,
               input_samples=itertools.repeat(Tns.sample1, 2),
               num_winners=2,
               neighbourhood_fn=None,
               presentation_duration=25,
               learning_rule=conditional_pca_learning,
               learning_rate=0.1,
               max_weight_value=Tns.max_weight)
    sum_weights_after_1st_epoch = \
        numpy.add.reduce(get_weights(Tns.prj1_2, Tns.max_weight)._weights, axis=0)
    weights_diff_1 = sum_weights_after_1st_epoch - sum_weights_before_1st_epoch
    argwinner_1 = numpy.argmax(weights_diff_1)
    # assert diff is all 0 except winners
    winner_diff_1 = weights_diff_1[argwinner_1]
    assert winner_diff_1 > 0
    weights_diff_1_nowin1 = numpy.array(weights_diff_1)
    weights_diff_1_nowin1[argwinner_1] -= winner_diff_1
    argwinner_2 = numpy.argmax(weights_diff_1_nowin1)
    winner_diff_2 = weights_diff_1[argwinner_2]
    assert winner_diff_2 > 0
    assert argwinner_1 != argwinner_2
    weights_diff_1_nowin2 = numpy.array(weights_diff_1_nowin1)
    weights_diff_1_nowin2[argwinner_2] -= winner_diff_2
    assert_allclose(weights_diff_1_nowin2, numpy.zeros(len(weights_diff_1)))
    # run some time without input to let the activity come back to 0
    from scheduling.pynn_scheduling import RATE_ENC_RESPAWN_DICT
    run_simulation(get_current_time()+30)
    # second epoch
    kwta_epoch(trained_population=Tns.p2,
               input_population=Tns.p1,
               projection=Tns.prj1_2,
               input_samples=itertools.repeat(Tns.sample2, 2),
               num_winners=2,
               neighbourhood_fn=None,
               presentation_duration=25,
               learning_rule=conditional_pca_learning,
               learning_rate=0.1,
               max_weight_value=Tns.max_weight)
    sum_weights_after_2nd_epoch = \
        numpy.add.reduce(get_weights(Tns.prj1_2, Tns.max_weight)._weights, axis=0)
    weights_diff_2 = sum_weights_after_2nd_epoch - sum_weights_after_1st_epoch
    argwinner_3 = numpy.argmax(weights_diff_2)
    assert argwinner_1 != argwinner_3 and argwinner_2 != argwinner_3
    # assert still only one non zero diff
    winner_diff_3 = weights_diff_2[argwinner_3]
    assert winner_diff_3 > 0
    weights_diff_2_nowin3 = numpy.array(weights_diff_2)
    weights_diff_2_nowin3[argwinner_3] -= winner_diff_3
    argwinner_4 = numpy.argmax(weights_diff_2_nowin3)
    winner_diff_4 = weights_diff_2[argwinner_4]
    assert winner_diff_4 > 0
    assert argwinner_1 != argwinner_4 and argwinner_2 != argwinner_4 \
        and argwinner_3 != argwinner_4
    weights_diff_2_nowin4 = numpy.array(weights_diff_2_nowin3)
    weights_diff_2_nowin4[argwinner_4] -= winner_diff_4
    assert_allclose(weights_diff_2_nowin4, numpy.zeros(len(weights_diff_2)))


def neighbourhood_f(pop, unit):
    l = r = t = b = (None, None, None)
    unit_index = pop.id_to_index(unit)
    u = pop.positions[0][unit_index], pop.positions[1][unit_index]
    d1, d2 = common.pynn_utils.rectilinear_shape(pop)
    for i in xrange(len(pop.positions[0])):
        x, y = pop.positions[0][i], pop.positions[1][i]
        if (x, y) == u:
            continue
        if x == u[0]:
            if y < u[1] and (t[1] < y or t[0] == None):
                t = x, y, i
            elif y > u[1] and (b[1] > y or b[0] == None):
                b = x, y, i
        elif y == u[1]:
            if x < u[0] and (l[0] < x or l[0] == None):
                l = x, y, i
            elif x > u[0] and (r[0] > x or r[0] == None):
                r = x, y, i
    return [(unit, 1)] + \
        [(pop[i], 0.5) for _, _, i in
         itertools.ifilter(lambda d: d[0] != None, [l, r, t, b])]


@with_setup(setup_input_samples)
@with_setup(setup_2_layers_ff_net)
def test_kwta_epoch_with_neighbourhood():
    Tns.p2.max_unit_rate=10
    Tns.p1.max_unit_rate=3.4
    sum_weights_before_1st_epoch = \
        numpy.add.reduce(get_weights(Tns.prj1_2, Tns.max_weight)._weights, axis=0)
    kwta_epoch(trained_population=Tns.p2,
               input_population=Tns.p1,
               projection=Tns.prj1_2,
               input_samples=itertools.repeat(Tns.sample1, 2),
               num_winners=1,
               neighbourhood_fn=neighbourhood_f,
               presentation_duration=25,
               learning_rule=conditional_pca_learning,
               learning_rate=0.1,
               max_weight_value=Tns.max_weight)
    sum_weights_after_1st_epoch = \
        numpy.add.reduce(get_weights(Tns.prj1_2, Tns.max_weight)._weights, axis=0)
    p2_shape = common.pynn_utils.rectilinear_shape(Tns.p2)
    weights_diff = sum_weights_after_1st_epoch - sum_weights_before_1st_epoch
    argwinner = numpy.argmax(weights_diff)
    num_neigh = 4
    x_pos = argwinner % p2_shape[1]
    if x_pos == 0 or x_pos == p2_shape[1]-1:
        num_neigh -= 1
    y_pos = argwinner / p2_shape[0]
    if y_pos == 0 or y_pos == p2_shape[0]-1:
        num_neigh -= 1
    for i in xrange(num_neigh+1):
        winner_diff = weights_diff[argwinner]
        assert winner_diff > 0
        weights_diff[argwinner] -= winner_diff
        argwinner = numpy.argmax(weights_diff)
    assert_allclose(weights_diff, numpy.zeros(len(weights_diff)))


@with_setup(setup_input_samples)
@with_setup(setup_2_layers_ff_net)
@raises(SimulationError)
def test_train_kwta_error_if_no_stop_condition():
    train_kwta(trained_population=Tns.p2,
               input_population=Tns.p1,
               projection=Tns.prj1_2,
               input_samples=itertools.repeat(Tns.sample1, 2),
               num_winners=1,
               neighbourhood_fn=neighbourhood_f,
               presentation_duration=25,
               learning_rule=conditional_pca_learning,
               learning_rate=.1,
               max_weight_value=Tns.max_weight,
               trained_pop_max_rate=None,
               input_pop_max_rate=None,
               min_delta_w=None,
               max_epoch=None)


@with_setup(setup_input_samples)
@with_setup(setup_2_layers_ff_net)
def test_train_kwta():
    Tns.p2.max_unit_rate=10
    Tns.p1.max_unit_rate=3.4
    sum_weights_before_1st_training = \
        numpy.add.reduce(get_weights(Tns.prj1_2, Tns.max_weight)._weights, axis=0)
    train_kwta(trained_population=Tns.p2,
               input_population=Tns.p1,
               projection=Tns.prj1_2,
               input_samples=list(itertools.repeat(Tns.sample1, 1)),
               num_winners=1,
               neighbourhood_fn=None,
               presentation_duration=22,
               learning_rule=conditional_pca_learning,
               learning_rate=.01,
               max_weight_value=Tns.max_weight,
               trained_pop_max_rate=None,
               input_pop_max_rate=None,
               min_delta_w=None,
               max_epoch=2)
    sum_weights_after_1st_training = \
        numpy.add.reduce(get_weights(Tns.prj1_2, Tns.max_weight)._weights, axis=0)
    weights_diff = sum_weights_after_1st_training - sum_weights_before_1st_training
    run_simulation(get_current_time()+30)
    train_kwta(trained_population=Tns.p2,
               input_population=Tns.p1,
               projection=Tns.prj1_2,
               input_samples=[Tns.sample2],
               num_winners=1,
               neighbourhood_fn=None,
               presentation_duration=22,
               learning_rule=conditional_pca_learning,
               learning_rate=0.5,
               max_weight_value=Tns.max_weight,
               trained_pop_max_rate=None,
               input_pop_max_rate=None,
               min_delta_w=0.05,
               max_epoch=None)
    sum_weights_after_2nd_training = \
        numpy.add.reduce(get_weights(Tns.prj1_2, Tns.max_weight)._weights, axis=0)
