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

from copy import deepcopy
import itertools
import logging
from logging import NullHandler 
from mock import Mock, MagicMock, patch
from nose import with_setup
from nose.tools import eq_, raises, timed, nottest
import os
import pyNN.brian as pynnn

import common.pynn_utils
from common.pynn_utils import *
from common.utils import splice
from numpy.testing import assert_allclose


class Tns(object):  # TestNameSpace
    pass

NAN = float("NaN")

CWD = os.getcwd()
BASE_DATA_TESTPATH = CWD[:CWD.rfind("/pycogmo")] + \
    "/pycogmo/tests/test_data/"
VALID_SAMPLE_INPUT_FILEPATHS = {
    "png"     : [BASE_DATA_TESTPATH + "bnw_checker_8x8_24bit.png",
                 BASE_DATA_TESTPATH + "bnw_checker_8x8_2bit.png", ],
    "colorpng": [BASE_DATA_TESTPATH + "color_checker_8x8_24bit_red.png",
                 BASE_DATA_TESTPATH + "color_checker_8x8_24bit_green.png",
                 BASE_DATA_TESTPATH + "color_checker_8x8_24bit_blue.png", ],
    "csv"     : [BASE_DATA_TESTPATH + "csv_checker.txt"]}
ALL_SAMPLE_INPUT_FILES = splice(VALID_SAMPLE_INPUT_FILEPATHS.values())

Tns.csv_checker_8x8_expected = \
Tns.png_checker_8x8_expected = [[1, 0] * 4, [0, 1] * 4] * 4


def list_units(ril):
    """Returns the list of PyNN units linked by the given rectilinear
    input layer."""
    return [b for _, b in
         list(splice(ril.unit_adapters_mat))]


def setup_weights():
    Tns.w0_array = []
    Tns.w0 = Weights(Tns.w0_array, 1)
    Tns.w1_array = [[j / 63. for j in range(i * 8, 8 + i * 8)]
                    for i in range(8)]
    Tns.w1 = Weights(Tns.w1_array, 1)
    Tns.w2_array = [[j / 63. for j in
                     list(itertools.chain(*zip(itertools.repeat(NAN),
                                               range(i * 8, 4 + i * 8))))]
                    for i in range(8)]
    Tns.w2 = Weights(Tns.w2_array, 1)
    Tns.w3_array = numpy.array(splice(
        [(list(itertools.chain(*zip(range(i * 8, 4 + i * 8), itertools.repeat(NAN)))),
          list(itertools.chain(*zip(itertools.repeat(NAN), range(i * 8, 4 + i * 8)))))
          for i in range(4)]), dtype=float) / 63.
    Tns.w3 = Weights(Tns.w3_array, 1)


def reset_pynn():
    pynnn.reset()


def setup_pynn_populations():
    pynnn.setup()
    Tns.p1 = pynnn.Population(64, pynnn.IF_curr_alpha,
                          structure=pynnn.space.Grid2D())
    Tns.p2 = pynnn.Population(64, pynnn.IF_curr_alpha,
                          structure=pynnn.space.Grid2D())
    Tns.prj1_2 = pynnn.Projection(
        Tns.p1, Tns.p2, pynnn.AllToAllConnector(allow_self_connections=False),
        target='excitatory')
    # Weights in nA as IF_curr_alpha uses current-based synapses
    Tns.prj1_2.set("weight", 1)
    Tns.max_weight = 33


def setup_mock_pynn_population():
    Tns.p_mock = MagicMock(spec=pynnn.Population)
    Tns.count_mock = Mock()
    Tns.p_mock.get_spike_counts.return_value = Tns.count_mock
    Tns.count_mock.get.return_value = 3
    Tns.p_mock.label = "Mock population"
    Tns.p_mock.size = 64
    Tns.p_mock.positions = [[0, 1, 2, 3, 4, 5, 6, 7] * 8] * 2


def setup_rectinilearinputlayers():
    setup_pynn_populations()
    Tns.ril1_max_namp = 1
    Tns.ril2_max_namp = 2
    Tns.ril1 = RectilinearInputLayer(Tns.p1, 8, 8, Tns.ril1_max_namp)
    Tns.ril2 = RectilinearInputLayer(Tns.p2, 8, 8, Tns.ril2_max_namp)


def setup_rectinilinear_ouput_rate_encoders():
    setup_pynn_populations()
    Tns.rore1_update_p = 22
    Tns.rore1_win_width = 200
    Tns.rore1_expected_h_len = 10
    Tns.rore2_update_p = 15
    Tns.rore2_win_width = 201
    Tns.rore2_expected_h_len = 14
    Tns.rore1 = RectilinearOutputRateEncoder(Tns.p1, 8, 8,
                                             Tns.rore1_update_p,
                                             Tns.rore1_win_width)
    Tns.rore2 = RectilinearOutputRateEncoder(Tns.p2, 8, 8,
                                             Tns.rore2_update_p,
                                             Tns.rore2_win_width)


def setup_registered_rectinilinear_ouput_rate_encoders():
    setup_rectinilinear_ouput_rate_encoders()
    common.pynn_utils.POP_ADAPT_DICT[(Tns.p1,
        common.pynn_utils.RectilinearOutputRateEncoder)] = Tns.rore1
    common.pynn_utils.POP_ADAPT_DICT[(Tns.p2,
        common.pynn_utils.RectilinearOutputRateEncoder)] = Tns.rore2


def test_InvalidFileFormatError():
    e = str(InvalidFileFormatError("test1", "test2"))
    assert "test1" in e
    assert "test2" in e


def test_InvalidMatrixShapeError():
    e = str(InvalidMatrixShapeError(1, 2, 3, 4))
    assert "1" in e
    assert "2" in e
    assert "3" in e
    assert "4" in e


@with_setup(setup_weights)
def test_weights__init__empty():
    "Test Weights.__init__ with an empty array"
    assert Tns.w0._dim1 == 0
    assert Tns.w0._dim2 == 0


@with_setup(setup_weights)
def test_weights_eq():
    assert Tns.w1_array == Tns.w1.non_normalized_weights, "initial data == internal data"
    assert Tns.w1.__eq__(numpy.array(Tns.w1_array)), "weights == numpy array"
    assert not Tns.w1.__eq__([1]), "length mismatch in dimension 1"
    assert Tns.w1.__eq__(Weights(Tns.w1_array, 1)), "only the instance changes"
    assert Tns.w2_array != Tns.w2.non_normalized_weights, "NaNs should not be equal"
    assert Tns.w2.__eq__(Weights(Tns.w2_array, 1)), "NaNs should be ignored"
    assert Tns.w1 != Tns.w2, "completely different objects, =="
    assert not Tns.w1.__eq__(Tns.w2), "completely different objects, __eq__"
    assert Tns.w0.__eq__([]), "empty weights == []"
    assert Tns.w0.__eq__(numpy.array([])), "empty weights == empty numpy array"
    assert not Tns.w0.__eq__(""), "incomparable classes"
    del Tns.w1_array[2][2]
    assert not Tns.w1.__eq__(numpy.array(Tns.w1_array)), \
        "length mismatch in dimension 2"


@with_setup(setup_weights)
def test_weights_shape():
    assert numpy.array(Tns.w1_array).shape == Tns.w1.shape


@with_setup(setup_weights)
def test_weights_accessors():
    assert (Tns.w1.non_normalized_numpy_weights == numpy.array(Tns.w1_array)).all()
    assert len(Tns.w1.flat_non_normalized_weights) == numpy.array(Tns.w1_array).size
    w = Weights([], 1)
    w.non_normalized_weights = numpy.array(Tns.w1_array)
    assert Tns.w1 == w
    assert Tns.w1[1][2] == Tns.w1_array[1][2]
    Tns.w1[1][2] = 1
    assert Tns.w1[1][2] == 1
    Tns.w1.non_normalized_weights = Tns.w2_array
    assert Tns.w1 == Tns.w2
    Tns.w1.non_normalized_weights = Tns.w1_array
    assert Tns.w1 == Tns.w1
    Tns.w1.non_normalized_weights = Tns.w2
    assert Tns.w1 == Tns.w2
    assert Tns.w2 == Weights(Tns.w2_array, 1)


@raises(TypeError)
@with_setup(setup_weights)
def test_weights_setter_raises_type_error():
    Tns.w1.non_normalized_weights = "1"


@raises(IndexError)
@with_setup(setup_weights)
def test_weights_get_item_exception_1():
    return Tns.w1[1000][2]


@raises(IndexError)
@with_setup(setup_weights)
def test_weights_get_item_exception_2():
    return Tns.w1[1][2000]


@raises(IndexError)
@with_setup(setup_weights)
def test_weights_set_item_exception_1():
    Tns.w1[1000][2] = 1


@raises(IndexError)
@with_setup(setup_weights)
def test_weights_set_item_exception_2():
    Tns.w1[1][2000] = 1


@with_setup(setup_weights)
def test_weights_set_item_1():
    Tns.w1[0][1] = .1337
    assert Tns.w1[0][1] == .1337


@with_setup(setup_weights)
def test_weights_set_item_2():
    Tns.w1.set_non_normalized_weight(1, 2, 2.)
    assert Tns.w1[1][2] == 2.


@with_setup(setup_weights)
def test_weights_get_normalized_weights_vector():
    for d in range(8):
        w1_to_d = Tns.w1.get_normalized_weights_vector(d)
        w2_to_d = Tns.w2.get_normalized_weights_vector(d)
        # with NaNs:
        if d % 2 == 0:
            assert not w2_to_d
        for s in range(8):
            assert w1_to_d[s] == Tns.w1_array[s][d]
            if d % 2 == 1:
                assert w2_to_d[s] == Tns.w2_array[s][d]


@with_setup(setup_weights)
def test_weights_set_normalized_weights_vector():
    Tns.w1.set_normalized_weights_vector(0, range(8))
    assert_allclose(Tns.w1.get_normalized_weights_vector(0), range(8))
    Tns.w2.set_normalized_weights_vector(2, [])
    assert_allclose(Tns.w2.get_normalized_weights_vector(2), [])
    Tns.w2.set_normalized_weights_vector(5, range(8))
    assert_allclose(Tns.w2.get_normalized_weights_vector(5), range(8))
    Tns.w3.set_normalized_weights_vector(7, range(4))
    assert_allclose(Tns.w3.get_normalized_weights_vector(7), range(4))


@raises(SimulationError)
@with_setup(setup_weights)
def test_weights_set_normalized_weights_vector_too_short_raises_error_1():
    Tns.w1.set_normalized_weights_vector(0, range(7))


@raises(SimulationError)
@with_setup(setup_weights)
def test_weights_set_normalized_weights_vector_too_short_raises_error_2():
    Tns.w2.set_normalized_weights_vector(1, [])


@raises(SimulationError)
@with_setup(setup_weights)
def test_weights_set_normalized_weights_vector_too_short_raises_error_3():
    Tns.w3.set_normalized_weights_vector(7, range(3))


@raises(SimulationError)
@with_setup(setup_weights)
def test_weights_set_normalized_weights_vector_too_long_raises_error_1():
    Tns.w1.set_normalized_weights_vector(1, range(9))


@raises(SimulationError)
@with_setup(setup_weights)
def test_weights_set_normalized_weights_vector_too_long_raises_error_2():
    Tns.w2.set_normalized_weights_vector(4, [1])


@raises(SimulationError)
@with_setup(setup_weights)
def test_weights_set_normalized_weights_vector_too_long_raises_error_3():
    Tns.w3.set_normalized_weights_vector(1, range(5))


@with_setup(setup_rectinilearinputlayers)
def test_get_weights():
    proj_w = Tns.prj1_2.getWeights(format='array')
    w = get_weights(Tns.prj1_2, 1.)
    assert w.shape == proj_w.shape
    assert (w.non_normalized_numpy_weights == proj_w).all()
    assert w._max_weight == 1.


@with_setup(setup_rectinilearinputlayers)
@with_setup(setup_weights)
def test_set_weights():
    w = Weights(numpy.zeros_like(Tns.prj1_2.getWeights(format='array')), 1)
    set_weights(Tns.prj1_2, w)
    proj_w = get_weights(Tns.prj1_2, 1).non_normalized_numpy_weights
    assert (proj_w == w.non_normalized_numpy_weights).all()


@raises(TypeError)
@with_setup(setup_rectinilearinputlayers)
@with_setup(setup_weights)
def test_set_weights_raises_TypeError():
    set_weights(Tns.prj1_2, Tns.w1_array)


def test_read_input_data_valid():
    for f in VALID_SAMPLE_INPUT_FILEPATHS["png"]:
        assert (read_input_data(f, 8, 8) ==
                Tns.png_checker_8x8_expected).all()
    for f in VALID_SAMPLE_INPUT_FILEPATHS["csv"]:
        assert (read_input_data(f, 8, 8) ==
                Tns.csv_checker_8x8_expected).all()
    for f in VALID_SAMPLE_INPUT_FILEPATHS["colorpng"]:
        m = read_input_data(f, 8, 8)
        t = m == Tns.csv_checker_8x8_expected
        t[0][0] = True
        assert t.all()
        assert m[0][0] < 1 and m[0][0] > 0


@raises(IOError)
def test_read_input_data_invalid_path():
    read_input_data("something invalid", 8, 8)


@raises(InvalidMatrixShapeError)
def test_read_input_data_incorrect_matrix_shape_1():
    read_input_data(VALID_SAMPLE_INPUT_FILEPATHS["png"][0], 4, 16)


@raises(InvalidMatrixShapeError)
def test_read_input_data_incorrect_matrix_shape_2():
    read_input_data(VALID_SAMPLE_INPUT_FILEPATHS["csv"][0], 4, 16)


@raises(InvalidFileFormatError)
def test_read_input_data_invalid_file_format_1():
    mm = Mock(magic.Magic)
    mm.from_file.return_value = "text/omgz"
    read_input_data(VALID_SAMPLE_INPUT_FILEPATHS["png"][0], 8, 8, m=mm)


@raises(InvalidFileFormatError)
def test_read_input_data_invalid_file_format_2():
    mm = Mock(magic.Magic)
    mm.from_file.return_value = "omgz/plain"
    read_input_data(VALID_SAMPLE_INPUT_FILEPATHS["png"][0], 8, 8, m=mm)


@raises(IOError)
def test_read_image_data_not_an_image():
    read_image_data(VALID_SAMPLE_INPUT_FILEPATHS["csv"][0])


@raises(IOError)
def test_read_csv_data_not_csv():
    read_csv_data(VALID_SAMPLE_INPUT_FILEPATHS["png"][0])


def test_verify_input_array_valid():
    verify_input_array(Tns.csv_checker_8x8_expected, 8, 8)


@raises(InvalidMatrixShapeError)
def test_verify_input_array_invalid_1():
    verify_input_array(Tns.csv_checker_8x8_expected, 8, 9)


@raises(InvalidMatrixShapeError)
def test_verify_input_array_invalid_2():
    verify_input_array(Tns.csv_checker_8x8_expected, 3, 8)


@raises(InvalidMatrixShapeError)
def test_verify_input_array_invalid_3():
    m = deepcopy(Tns.csv_checker_8x8_expected)
    m[3] = [1] * 9
    verify_input_array(m, 8, 8)


@raises(TypeError)
def test_verify_input_array_invalid_4():
    m = deepcopy(Tns.csv_checker_8x8_expected)
    m[3][3] = "a"
    verify_input_array(m, 8, 8)


def mock_read_input_data_setup():
    import common.pynn_utils
    Tns.rid_patcher = patch("common.pynn_utils.read_input_data")
    Tns.rid_mock = Tns.rid_patcher.start()
    Tns.rid_mock.return_value = [[1]]


def mock_read_input_data_teardown():
    Tns.rid_patcher.stop()
    del Tns.rid_mock


@with_setup(mock_read_input_data_setup, mock_read_input_data_teardown)
def test_input_sample_init_string_param_calls_read_input_data():
    import common.pynn_utils
    path = "example"
    InputSample(1, 1, path)
    common.pynn_utils.read_input_data.assert_called_once_with(path, 1, 1)


@raises(IOError)
def test_input_sample_init_string_param_IO_error_propagates():
    InputSample(1, 1, "I bet this will never be a filename in my test")


@raises(TypeError)
def test_input_sample_init_fileobj_param_raises_type_error():
    with open(VALID_SAMPLE_INPUT_FILEPATHS["csv"][0], 'rb') as f:
        InputSample(1, 1, f)


def test_input_sample_init_list_param():
    s = InputSample(1, 1, [[1]])
    assert s._array == [[1]]


def test_input_sample_init_getitem_expand_param():
    mock_obj = Mock()
    mock_obj.__getitem__ = Mock(return_value=[0, 1, 2])
    mock_obj.__setitem__ = Mock()
    s = InputSample(4, 3, mock_obj, expand=True)
    assert s._array == [[0, 1, 2]] * 4
    expected = [((0,), {})] * 3 + \
        [((1,), {})] * 3 + [((2,), {})] * 3 + [((3,), {})] * 3
    assert mock_obj.__getitem__.call_args_list == expected
    s[3] = ["s"]
    assert s._setitem == s._assign_to_array
    assert not mock_obj.__setitem__.called  # expand=T copied the contents


def test_input_sample_init_getitem_noexpand_param():
    mock_obj = Mock()
    mock_obj.__getitem__ = Mock(return_value=[0, 1, 2])
    mock_obj.__setitem__ = Mock()
    s = InputSample(4, 3, mock_obj, expand=False)
    assert hasattr(s, '__getitem__')
    assert s[1] == [0, 1, 2]
    mock_obj.__getitem__.assert_called_once_with(1)


@raises(TypeError)
def test_input_sample_init_getitem_noexpand_param_setitem_raises_typerr():
    mock_obj = Mock()
    mock_obj.__getitem__ = Mock(return_value=[0, 1, 2])
    s = InputSample(4, 3, mock_obj, expand=False)
    s[3] = ["s"]


def test_input_sample_init_callable_expand_param():
    mock_obj = Mock(return_value=2)
    s = InputSample(4, 3, mock_obj, expand=True)
    assert s._array == [[2] * 3] * 4
    expected = zip(itertools.product(range(4), range(3)), itertools.repeat({}))
    assert mock_obj.call_args_list == expected
    mock_obj.reset_mock()
    mock_obj.__setitem__ = Mock()
    s[3] = ["s"]
    assert s._setitem == s._assign_to_array
    assert not mock_obj.__setitem__.called


def test_input_sample_init_callable_noexpand_param():
    mock_obj = Mock(return_value=2)
    s = InputSample(4, 3, mock_obj, expand=False)
    assert hasattr(s[3], '__getitem__')
    assert s[3][4] == 2
    mock_obj.assert_called_once_with(3, 4)


@raises(TypeError)
def test_input_sample_init_callable_noexpand_param_setitem_raises_typerr():
    mock_obj = Mock(return_value=2)
    s = InputSample(4, 3, mock_obj, expand=False)
    s[3] = ["s"]


def test_input_sample_access_and_mod_real_file():
    s = InputSample(8, 8, VALID_SAMPLE_INPUT_FILEPATHS['png'][0])
    for i, l in enumerate(Tns.png_checker_8x8_expected):
        for j, c in enumerate(l):
            assert s[i][j] == c
    s[7][7] = 0.5
    assert s[7][7] == 0.5


@with_setup(setup_pynn_populations)
def test_rectilinear_layer_adapter__init():
    rla = RectilinearLayerAdapter(Tns.p1, 8, 8)
    assert rla.pynn_population == Tns.p1


@with_setup(setup_pynn_populations)
def test_rectilinear_layer_adapter_shape():
    rla = RectilinearLayerAdapter(Tns.p1, 8, 8)
    assert rla.shape == (8, 8)
    rla = RectilinearLayerAdapter(Tns.p1, 4, 16)
    assert rla.shape == (4, 16)


@with_setup(setup_pynn_populations)
def test_rectilinear_layer_adapter__getitem():
    rla = RectilinearLayerAdapter(Tns.p1, 8, 8)
    rla2 = RectilinearLayerAdapter(Tns.p1, 4, 16)
    for (i, j) in itertools.product(xrange(8), repeat=2):
        assert rla[i][j][1] == Tns.p1[i * 8 + j]
    for (i, j) in itertools.product(xrange(4), xrange(16)):
        assert rla2[i][j][1] == Tns.p1[i * 16 + j]


@raises(IndexError)
@with_setup(setup_pynn_populations)
def test_rectilinear_layer_adapter__getitem_out_of_bounds_1():
    rla = RectilinearLayerAdapter(Tns.p1, 8, 8)
    rla[1][12]


@raises(IndexError)
@with_setup(setup_pynn_populations)
def test_rectilinear_layer_adapter__getitem_out_of_bounds_2():
    rla = RectilinearLayerAdapter(Tns.p1, 8, 8)
    rla[12][1]


@with_setup(setup_pynn_populations)
def test_rectilinear_layer_adapter_get_unit():
    rla = RectilinearLayerAdapter(Tns.p1, 8, 8)
    rla2 = RectilinearLayerAdapter(Tns.p1, 4, 16)
    for (i, j) in itertools.product(xrange(8), repeat=2):
        assert rla.get_unit(i, j) == Tns.p1[i * 8 + j]
    for (i, j) in itertools.product(xrange(4), xrange(16)):
        assert rla2.get_unit(i, j) == Tns.p1[i * 16 + j]


@with_setup(setup_rectinilearinputlayers)
def test_rectilinear_input_layer__init():
    assert Tns.ril1.input_scaling == Tns.ril1_max_namp
    assert Tns.ril2.input_scaling == Tns.ril2_max_namp
    assert RectilinearInputLayer(Tns.p1, 8, 8).input_scaling == \
        INPUT_LAYER_MAX_NAMP_DEFAULT
    assert len(Tns.ril1.unit_adapters_mat) == 8
    assert set(Tns.p1) == set(list_units(Tns.ril1))
    assert set(Tns.p2) == set(list_units(Tns.ril2))


@with_setup(setup_rectinilearinputlayers)
def test_rectilinear_input_layer_access():
    a, b = Tns.ril1[1][2]  # __getitem__
    assert a == None
    assert b in set(Tns.p1)
    assert Tns.ril1.shape == (8, 8)


@with_setup(setup_rectinilearinputlayers)
def test_rectilinear_input_layer_apply_input():
    mock_dcsource = Mock(spec=pynnn.DCSource)
    some_sample = [[5] * 8] * 8
    Tns.ril1.apply_input(sample=some_sample,
                         start_time=12, duration=51,
                         max_namp=None, dcsource_class=mock_dcsource)
    assert mock_dcsource.call_count == 64
    for args_i in itertools.product(xrange(8), repeat=2):
        args = mock_dcsource.call_args_list[args_i[0] * 8 + args_i[1]]
        assert args[0][0] == {'amplitude': Tns.ril1_max_namp * \
                               some_sample[args_i[0]][args_i[1]],
                           'start': 12, 'stop': 12 + 51}


@with_setup(setup_rectinilinear_ouput_rate_encoders)
def test_rectilinear_ouput_rate_encoder__init():
    assert Tns.rore1.hist_len == Tns.rore1_expected_h_len
    assert Tns.rore2.hist_len == Tns.rore2_expected_h_len


@with_setup(setup_rectinilinear_ouput_rate_encoders)
def test_rectilinear_ouput_rate_encoder_extend_capacity():
    Tns.rore1.pynn_population = MagicMock()
    Tns.rore1.update_rates(1)
    l1 = len(Tns.rore1.unit_adapters_mat[0][0][0])
    l2 = Tns.rore1.window_width
    l3 = len(Tns.rore1.update_history)
    l4 = Tns.rore1.hist_len
    Tns.rore1.extend_capacity(0)
    assert l1 == len(Tns.rore1.unit_adapters_mat[0][0][0]) - 1
    assert l2 == Tns.rore1.window_width
    assert l3 == len(Tns.rore1.update_history) - 1
    assert l4 == Tns.rore1.hist_len - 1
    Tns.rore1.extend_capacity(Tns.rore1.hist_len-1)
    assert l1 == len(Tns.rore1.unit_adapters_mat[0][0][0]) - 2
    assert l2 == Tns.rore1.window_width
    assert l3 == len(Tns.rore1.update_history) - 2
    assert l4 == Tns.rore1.hist_len - 2


@with_setup(setup_rectinilinear_ouput_rate_encoders)
def test_rectilinear_ouput_rate_encoder_make_hist_weights_vec():
    wv0 = Tns.rore1.make_hist_weights_vec([200, -50, 0, 50, 100, 150], 200, 0)
    wv0_ans = [0., 0.0625, 0.1875, 0.3125, 0.4375]
    wv1 = Tns.rore1.make_hist_weights_vec(
        [190, 200, 0, 120, 140, 160], 110, 1)
    wv2 = Tns.rore1.make_hist_weights_vec(
        [190, 200, 95, 120, 140, 160], 40, 1)
    assert (wv0 == Tns.rore1.make_hist_weights_vec(
        [200, -50, 0, 50, 100, 150], 200, 0)).all()
    assert (wv0_ans == wv0).all()
    assert sum(wv0) == 1.
    assert sum(wv1) == 1.
    assert sum(wv2) == 1.
    assert (wv2[0:3] == numpy.zeros(3)).all()


@with_setup(setup_rectinilinear_ouput_rate_encoders)
def test_rectilinear_ouput_rate_encoder_advance_idx_and_previous_idx():
    for i in xrange(Tns.rore1_expected_h_len):
        previous_idx = Tns.rore1.idx
        Tns.rore1.advance_idx()
        assert Tns.rore1.idx == i
        if previous_idx != -1:
            assert Tns.rore1.previous_idx == previous_idx
    Tns.rore1.advance_idx()
    assert Tns.rore1.idx == 0
    assert Tns.rore1.previous_idx == Tns.rore1_expected_h_len - 1


@with_setup(setup_rectinilinear_ouput_rate_encoders)
@with_setup(setup_mock_pynn_population)
def test_rectilinear_ouput_rate_encoder_update_rates_and_get_rates():
    rore = RectilinearOutputRateEncoder(Tns.p_mock, 8, 8,
                                        Tns.rore1_update_p,  # 22
                                        Tns.rore1_win_width) # 200
    Tns.a_counter = 0
    def inc_by_3(*args):
        Tns.a_counter += 1
        return Tns.a_counter / 64 * 3
    Tns.count_mock.get = inc_by_3
    for i in range(Tns.rore1_win_width / Tns.rore1_update_p + 1):
        rore.update_rates(i*22)
    Tns.count_mock.assert_called()
    for r in splice(rore.get_rates()):
        assert_allclose(r, 3.0 / 22, atol=0.0001)


@with_setup(setup_rectinilinear_ouput_rate_encoders)
@with_setup(setup_mock_pynn_population)
def test_rectilinear_ouput_rate_encoder_update_rates_with_irregular_period():
    rore = RectilinearOutputRateEncoder(Tns.p_mock, 8, 8,
                                        Tns.rore1_update_p,  # 22
                                        Tns.rore1_win_width) # 200
    Tns.a_counter = 0
    def inc_by_3_or_6(*args):
        cycle = Tns.a_counter / 64
        Tns.a_counter += 1
        return cycle / 2 * 6 + (cycle + 1) / 2 * 3
    Tns.count_mock.get = inc_by_3_or_6
    for i in range(2 * Tns.rore1_win_width / Tns.rore1_update_p + 1):
        # 0 -> 0, 1 -> 11, 2 -> 33, 3 -> 44
        rore.update_rates(i / 2 * 22 + (i + 1) / 2 * 11)
    Tns.count_mock.assert_called()
    for r in splice(rore.get_rates()):
        assert_allclose(r, 9.0 / 33, atol=0.0001)


@with_setup(setup_rectinilinear_ouput_rate_encoders)
@with_setup(setup_mock_pynn_population)
@raises(SimulationError)
def test_rectilinear_ouput_rate_encoder_update_rates_with_nonmonotonic_time():
    rore = RectilinearOutputRateEncoder(Tns.p_mock, 8, 8,
                                        Tns.rore1_update_p,
                                        Tns.rore1_win_width)
    Tns.count_mock.get = lambda _: 3
    for i in range(5):
        rore.update_rates(i % 4)


@with_setup(setup_mock_pynn_population)
def test_rectilinear_ouput_rate_encoder_f_rate():
    rore = RectilinearOutputRateEncoder(Tns.p_mock, dim1=8, dim2=8,
                                        update_period=50, window_width=200)
    np_a = numpy.array([4, 5, 10, 0, 0, 0])
    result_shift_3 = numpy.array([0., 0.0625, 0.1875, 0.3125, 0.4375]).dot(
        [0, 0, 4/50., 1/50., 5/50.])
    rore.advance_idx()
    rore.advance_idx()
    rore.advance_idx()
    rates = rore.f_rate(np_a, update_history=[100, 150, 200, -50, 0, 50])
    assert_allclose(rates, result_shift_3)


@with_setup(setup_mock_pynn_population)
def test_population_adapter_provider():
    d1 = {}
    provided1 = population_adpater_provider(d1, RectilinearLayerAdapter,
                                            Tns.p_mock)
    assert provided1.shape == (8, 8)
    assert provided1.pynn_population == Tns.p_mock
    provided2 = RectilinearLayerAdapter(Tns.p_mock, 8, 8)
    d2 = {(Tns.p_mock, RectilinearLayerAdapter): provided2}
    assert population_adpater_provider(d2, RectilinearLayerAdapter,
                                       Tns.p_mock) == provided2


@raises(TypeError)
def test_population_adapter_provider__non_square_pop_raises_type_error():
    pop1 = Mock()
    pop1.size = 10
    population_adpater_provider(POP_ADAPT_DICT, Mock, pop1)


@with_setup(setup_pynn_populations)
def test_get_input_layer():
    p1i = get_input_layer(Tns.p1)
    assert get_input_layer(Tns.p1) == p1i
    assert get_input_layer(Tns.p2) != p1i
    from common.pynn_utils import POP_ADAPT_DICT as d
    assert (Tns.p1, RectilinearInputLayer) in d.keys()
    assert (Tns.p2, RectilinearInputLayer) in d.keys()


@with_setup(setup_pynn_populations)
def test_get_rate_encoder():
    re = get_rate_encoder(Tns.p1)
    assert get_rate_encoder(Tns.p1) == re
    assert get_rate_encoder(Tns.p2) != re
    from common.pynn_utils import POP_ADAPT_DICT as d
    assert (Tns.p1, RectilinearOutputRateEncoder) in d.keys()
    assert (Tns.p2, RectilinearOutputRateEncoder) in d.keys()


def setup_pynn_populations_with_full_connectivity():
    pynnn.setup()
    Tns.p1 = pynnn.Population(4, pynnn.IF_curr_alpha,
                          structure=pynnn.space.Grid2D())
    Tns.p2 = pynnn.Population(4, pynnn.IF_curr_alpha,
                          structure=pynnn.space.Grid2D())
    Tns.prj1_2 = pynnn.Projection(
        Tns.p1, Tns.p2, pynnn.AllToAllConnector(allow_self_connections=False),
        target='excitatory')


def setup_rectinilinear_ouput_rate_encoders_full():
    setup_pynn_populations_with_full_connectivity()
    Tns.rore1_update_p = 22
    Tns.rore1_win_width = 200
    Tns.rore1_expected_h_len = 10
    Tns.rore2_update_p = 15
    Tns.rore2_win_width = 201
    Tns.rore2_expected_h_len = 14
    Tns.rore1 = RectilinearOutputRateEncoder(Tns.p1, 2, 2,
                                             Tns.rore1_update_p,
                                             Tns.rore1_win_width)
    Tns.rore2 = RectilinearOutputRateEncoder(Tns.p2, 2, 2,
                                             Tns.rore2_update_p,
                                             Tns.rore2_win_width)


def setup_mock_rectinilinear_ouput_rate_encoders_full():
    setup_rectinilinear_ouput_rate_encoders_full()
    Tns.rore1.idx = 5
    r = numpy.array(range(2), dtype=numpy.float)
    Tns.r = numpy.array([r, r+2],
                    dtype=numpy.float)
    Tns.rore1.get_rate = lambda x, y, t : Tns.r[x][y]
    common.pynn_utils.POP_ADAPT_DICT[(Tns.p1,
        common.pynn_utils.RectilinearOutputRateEncoder)] = Tns.rore1
    common.pynn_utils.POP_ADAPT_DICT[(Tns.p2,
        common.pynn_utils.RectilinearOutputRateEncoder)] = Tns.rore2


def setup_pynn_populations_with_1_to_1_connectivity():
    pynnn.setup()
    Tns.p1 = pynnn.Population(64, pynnn.IF_curr_alpha,
                          structure=pynnn.space.Grid2D())
    Tns.p2 = pynnn.Population(64, pynnn.IF_curr_alpha,
                          structure=pynnn.space.Grid2D())
    Tns.prj1_2 = pynnn.Projection(
        Tns.p1, Tns.p2, pynnn.OneToOneConnector(),
        target='excitatory')


def setup_rectinilinear_ouput_rate_encoders_1_to_1():
    setup_pynn_populations_with_1_to_1_connectivity()
    Tns.rore1_update_p = 22
    Tns.rore1_win_width = 200
    Tns.rore1_expected_h_len = 10
    Tns.rore2_update_p = 15
    Tns.rore2_win_width = 201
    Tns.rore2_expected_h_len = 14
    Tns.rore1 = RectilinearOutputRateEncoder(Tns.p1, 8, 8,
                                             Tns.rore1_update_p,
                                             Tns.rore1_win_width)
    Tns.rore2 = RectilinearOutputRateEncoder(Tns.p2, 8, 8,
                                             Tns.rore2_update_p,
                                             Tns.rore2_win_width)


def setup_mock_rectinilinear_ouput_rate_encoders_1_to_1():
    setup_rectinilinear_ouput_rate_encoders_1_to_1()
    Tns.rore1.idx = 5
    r = numpy.array(range(8), dtype=numpy.float)
    Tns.r = numpy.array([r, r+8, r+16, r+24, r+32, r+40, r+48, r+56],
                    dtype=numpy.float)
    Tns.rore1.get_rate = lambda x, y, t : Tns.r[x][y]
    common.pynn_utils.POP_ADAPT_DICT[(Tns.p1,
        common.pynn_utils.RectilinearOutputRateEncoder)] = Tns.rore1
    common.pynn_utils.POP_ADAPT_DICT[(Tns.p2,
        common.pynn_utils.RectilinearOutputRateEncoder)] = Tns.rore2


@with_setup(setup_mock_rectinilinear_ouput_rate_encoders_full)
def test_presynaptic_outputs_full_connectivity():
    for u in Tns.p2:
        assert_allclose(presynaptic_outputs(u, Tns.prj1_2), splice(Tns.r))


@with_setup(setup_mock_rectinilinear_ouput_rate_encoders_1_to_1)
def test_presynaptic_outputs_1_to_1_connectivity():
    for u in Tns.p2:
        assert_allclose(presynaptic_outputs(u, Tns.prj1_2),
                        [splice(Tns.r)[Tns.p2.id_to_index(u)]])

