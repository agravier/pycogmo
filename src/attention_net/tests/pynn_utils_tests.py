#!/usr/bin/env python2

from copy import deepcopy
import itertools
import logging
from logging import NullHandler 
from mock import Mock, patch
from nose import with_setup
from nose.tools import eq_, raises, timed, nottest
import os
import pyNN.nest as pynnn
from common.pynn_utils import *
from common.utils import splice

class Tns(object): # TestNameSpace
    pass

NAN = float("NaN")

CWD = os.getcwd()
BASE_DATA_TESTPATH = CWD[:CWD.rfind("/src")] + \
    "/src/attention_net/tests/test_data/"
VALID_SAMPLE_INPUT_FILEPATHS = {
    "png"     : [BASE_DATA_TESTPATH + "bnw_checker_8x8_24bit.png",
                 BASE_DATA_TESTPATH + "bnw_checker_8x8_2bit.png",],
    "colorpng": [BASE_DATA_TESTPATH + "color_checker_8x8_24bit_red.png",
                 BASE_DATA_TESTPATH + "color_checker_8x8_24bit_green.png",
                 BASE_DATA_TESTPATH + "color_checker_8x8_24bit_blue.png",],
    "csv"     : [BASE_DATA_TESTPATH + "csv_checker.txt"]}
ALL_SAMPLE_INPUT_FILES = splice(VALID_SAMPLE_INPUT_FILEPATHS.values())

Tns.csv_checker_8x8_expected = \
Tns.png_checker_8x8_expected = [[1,0]*4,[0,1]*4]*4

def list_units(ril):
    """Returns the list of PyNN units linked by the given rectilinear
    input layer."""
    return [b for a, b in
         list(splice(ril.electrodes))]

def setup_weights():
    Tns.w1_array = [[j/63. for j in range(i*8,8+i*8)] 
                    for i in range(8)]
    Tns.w1 = Weights(Tns.w1_array)
    Tns.w2_array = [[j/63. for j in 
                     list(itertools.chain(*zip(itertools.repeat(NAN), 
                                               range(i*8,4+i*8))))]
                    for i in range(8)]
    Tns.w2 = Weights(Tns.w2_array)

def setup_pynn_populations():
    pynnn.setup()
    Tns.p1 = pynnn.Population(64, pynnn.IF_curr_alpha,
                          structure=pynnn.space.Grid2D())
    Tns.p2 = pynnn.Population(64, pynnn.IF_curr_alpha,
                          structure=pynnn.space.Grid2D())


def setup_rectinilearinputlayers():
    setup_pynn_populations()
    Tns.ril1_max_namp = 1
    Tns.ril2_max_namp = 2
    Tns.ril1 = RectilinearInputLayer(Tns.p1, 8, 8, Tns.ril1_max_namp)
    Tns.ril2 = RectilinearInputLayer(Tns.p2, 8, 8, Tns.ril2_max_namp)

# def setup_mock_dcsource():
#     Tns.dcsource_patch = patch.object(pynnn., "__int__")
#     Tns.dcsource_patch.start()

# def teardown_mock_dcsource():
#     Tns.dcsource_patch.stop()
#     Tns.dcsource_patch = None

@with_setup(setup_weights)
def test_weights_eq():
    assert Tns.w1_array == Tns.w1.weights, "initial data == property"
    assert Tns.w1 == Weights(Tns.w1_array), "only the instance changes"
    assert Tns.w2_array != Tns.w2.weights, "NaNs should not be equal" # Because of NaNs
    assert Tns.w2 == Weights(Tns.w2_array), "NaNs should be ignored"
    assert Tns.w1 != Tns.w2, "completetly different objects"

@with_setup(setup_weights)
def test_weights_accessors():
    assert Tns.w1[1][2] == Tns.w1_array[1][2]
    Tns.w1[1][2] = 1
    assert Tns.w1[1][2] == 1
    Tns.w1.weights = Tns.w2_array
    assert Tns.w1 == Tns.w2
    Tns.w1.weights = Tns.w1_array
    assert Tns.w1 == Tns.w1
    Tns.w1.weights = Tns.w2
    assert Tns.w1 == Tns.w2
    assert Tns.w2 == Weights(Tns.w2_array)

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
def test_weights_adjusted():
    error = deepcopy(Tns.w1_array)
    assert Tns.w2 == Tns.w2.adjusted(error, learning=0)

    r1 = Tns.w1.adjusted(error, learning=1)
    for i, j in itertools.product(xrange(8), repeat=2):
        error[i][j] = 0
    assert r1 == error
    
    assert Tns.w2.adjusted(error) == Tns.w2

    for i, j in itertools.product(xrange(8), repeat=2):
        error[i][j] = 1
        Tns.w2_array[i][j] = Tns.w2_array[i][j] - Tns.w2._default_l_rate

    assert Tns.w2.adjusted(error) == Weights(Tns.w2_array)

@with_setup(setup_rectinilearinputlayers)
def test_rectilinearinputlayer_init():
    assert len(Tns.ril1.electrodes) == 8
    assert set(Tns.p1) == set(list_units(Tns.ril1))
    assert set(Tns.p2) == set(list_units(Tns.ril2))

@with_setup(setup_rectinilearinputlayers)
def test_rectilinearinputlayer_access():
    a, b = Tns.ril1[1][2] # __getitem__
    assert a == None
    assert b in set(Tns.p1)
    assert Tns.ril1.shape == (8, 8)

@with_setup(setup_rectinilearinputlayers)
def test_rectilinearinputlayer_apply_input():
    mock_dcsource = Mock(spec = pynnn.DCSource)
    some_sample = [[5]*8]*8
    Tns.ril1.apply_input(sample = some_sample, 
                         start_time = 12, duration = 51,
                         max_namp = None, dcsource_class = mock_dcsource)
    assert mock_dcsource.call_count == 64
    for args_i in itertools.product(xrange(8), repeat=2):
        args = mock_dcsource.call_args_list[args_i[0]*8+args_i[1]]
        assert args[0] == ()
        assert args[1] == {'amplitude' : Tns.ril1_max_namp * \
                               some_sample[args_i[0]][args_i[1]] , 
                           'start' : 12, 'stop' : 12 + 51}


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
    m[3] = [1]*9
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
    Tns.rid_mock.return_value=[[1]]

def mock_read_input_data_teardown():
    Tns.rid_patcher.stop()
    del Tns.rid_mock

@with_setup(mock_read_input_data_setup, mock_read_input_data_teardown)
def test_input_sample_init_string_param_calls_read_input_data():
    import common.pynn_utils
    path = "example"
    InputSample(1, 1, path)
    common.pynn_utils.read_input_data.assert_called_once_with(path, 1, 1)

@raises(TypeError)
def test_input_sample_init_fileobj_param_raises_type_error():
    with open(VALID_SAMPLE_INPUT_FILEPATHS["csv"][0], 'rb') as f:
        InputSample(1, 1, f)

def test_input_sample_init_list_param():
    s = InputSample(1, 1, [[1]])
    assert s._array == [[1]]

def test_input_sample_init_getitem_expand_param():
    mock_obj = Mock()
    mock_obj.__getitem__ = Mock(return_value=[0,1,2])
    mock_obj.__setitem__ = Mock()
    s = InputSample(4, 3, mock_obj, expand=True)
    assert s._array == [[0,1,2]]*4
    expected = [((0,),{})]*3+[((1,),{})]*3+[((2,),{})]*3+[((3,),{})]*3
    assert mock_obj.__getitem__.call_args_list == expected
    s[3] = ["s"]
    assert s._setitem == s._assign_to_array
    assert not mock_obj.__setitem__.called # expand=T copied the contents

def test_input_sample_init_getitem_noexpand_param():
    mock_obj = Mock()
    mock_obj.__getitem__ = Mock(return_value=[0,1,2])
    mock_obj.__setitem__ = Mock()
    s = InputSample(4, 3, mock_obj, expand=False)
    assert hasattr(s, '__getitem__')
    assert s[1] == [0,1,2]
    mock_obj.__getitem__.assert_called_once_with(1)

@raises(TypeError)
def test_input_sample_init_getitem_noexpand_param_setitem_raises_typerr():
    mock_obj = Mock()
    mock_obj.__getitem__ = Mock(return_value=[0,1,2])
    s = InputSample(4, 3, mock_obj, expand=False)
    s[3] = ["s"]

def test_input_sample_init_callable_expand_param():
    mock_obj = Mock(return_value=2)
    s = InputSample(4, 3, mock_obj, expand=True)
    assert s._array == [[2]*3]*4
    expected = zip(itertools.product(range(4),range(3)), itertools.repeat({}))
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
