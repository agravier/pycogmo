#!/usr/bin/env python2

import itertools
import logging
from logging import NullHandler 
from mock import Mock, patch
from nose import with_setup
from nose.tools import eq_, raises, timed, nottest
from common.pynn_utils import *

NAN = float("NaN")

class Tns(object):
    pass

def setup_weights():
    Tns.w1_array = [[j/63. for j in range(i*8,8+i*8)] 
                    for i in range(8)]
    Tns.w1 = Weights(Tns.w1_array)
    Tns.w2_array = [[j/63. for j in 
                     list(itertools.chain(*zip(itertools.repeat(NAN), 
                                               range(i*8,4+i*8))))]
                    for i in range(8)]
    Tns.w2 = Weights(Tns.w2_array)

@with_setup(setup_weights)
def test_weights():
    assert Tns.w1_array == Tns.w1.weights, "initial data == property"
    assert Tns.w1 == Weights(Tns.w1_array), "only the instance changes"
    assert Tns.w2_array != Tns.w2.weights, "NaNs should not be equal" # Because of NaNs
    assert Tns.w2 == Weights(Tns.w2_array), "NaNs should be ignored"
    assert Tns.w1 != Tns.w2, "completetly different objects"

