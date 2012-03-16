#!/usr/bin/env python2

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
from numpy.testing.utils import assert_almost_equal

DUMMY_LOGGER = logging.getLogger("testLogger")
DUMMY_LOGGER.addHandler(NullHandler())

# TODO Check for runaway positive feedback of SOM units


class Tns:
    pass


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
    assert_almost_equal(oja_learning(Tns.s_in_1, Tns.s_out_1, Tns.s_w_1, 1),
                        0.72)
    assert oja_learning(Tns.s_in_2, Tns.s_out_2, Tns.s_w_2, 0.5) == Tns.s_w_2
    assert_almost_equal(oja_learning(Tns.s_in_3, Tns.s_out_3, Tns.s_w_3, 3.5),
                        -1)


@with_setup(setup_data)
def test_oja_learning_vector():
    assert len(oja_learning(Tns.v_in_1, Tns.s_out_1, Tns.v_w_1, 1)) == 5
    assert_almost_equal(oja_learning(Tns.v_in_1, Tns.s_out_1, Tns.v_w_1, 1),
        [0., 0.919, 0.057, 0.313, 0.64])
    assert len(oja_learning(Tns.v_in_2, Tns.s_out_2, Tns.v_w_2, 0.5)) == 5
    assert (oja_learning(Tns.v_in_2, Tns.s_out_2, Tns.v_w_2, 0.5) == \
        [-13., 0.4, -1., 2., 5.]).all()
    assert len(oja_learning(Tns.v_in_3, Tns.s_out_3, Tns.v_w_3, 3.5)) == 5
    assert (oja_learning(Tns.v_in_3, Tns.s_out_3, Tns.v_w_3, 3.5) == \
        [-6] * 5).all()


@with_setup(setup_data)
def test_conditional_pca_learning_scalar():
    assert_almost_equal(conditional_pca_learning(Tns.s_in_1, Tns.s_out_1,
        Tns.s_w_1, 1), 0.72)
    assert conditional_pca_learning(Tns.s_in_2, Tns.s_out_2, Tns.s_w_2, 0.5) \
        == Tns.s_w_2
    assert_almost_equal(conditional_pca_learning(Tns.s_in_3, Tns.s_out_3,
        Tns.s_w_3, 3.5), 6.7)


@with_setup(setup_data)
def test_conditional_pca_learning_vector():
    assert len(conditional_pca_learning(Tns.v_in_1, Tns.s_out_1, \
        Tns.v_w_1, 1)) == 5
    assert_almost_equal(conditional_pca_learning(Tns.v_in_1, Tns.s_out_1, \
        Tns.v_w_1, 1), [0., 0.91, 0.03, 0.25, 0.55])
    assert len(conditional_pca_learning(Tns.v_in_2, Tns.s_out_2, \
        Tns.v_w_2, 0.5)) == 5
    assert (conditional_pca_learning(Tns.v_in_2, Tns.s_out_2, Tns.v_w_2, 0.5) \
        == [-13., 0.4, -1., 2., 5.]).all()
    assert len(conditional_pca_learning(Tns.v_in_3, Tns.s_out_3, \
        Tns.v_w_3, 3.5)) == 5
    assert (conditional_pca_learning(Tns.v_in_3, Tns.s_out_3, Tns.v_w_3, 3.5) \
            == [1.] * 5).all()
