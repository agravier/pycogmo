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

import os
import itertools
import logging
from logging import NullHandler 
from mock import Mock, MagicMock, patch
from nose import with_setup
from nose.tools import eq_, raises, timed, nottest
import os

from common.utils import *

class Tns(object):  # TestNameSpace
    pass


class MockLoggingHandler(logging.Handler):
    """Mock logging handler to check for expected logs."""

    def __init__(self, *args, **kwargs):
        self.reset()
        logging.Handler.__init__(self, *args, **kwargs)

    def emit(self, record):
        self.messages[record.levelname.lower()].append(record.getMessage())

    def reset(self):
        self.messages = {
            'debug': [],
            'info': [],
            'warning': [],
            'error': [],
            'critical': [],
        }


def test_configure_loggers():
    l = logging.getLogger("test_logger")
    assert l.handlers == [], "The test logger existed before the test"
    mdh, mfh = Mock(), Mock()
    configure_loggers(mdh, mfh, l)
    assert mdh in l.handlers
    assert mfh in l.handlers


def test_optimal_rounding():
    assert optimal_rounding(1.002) == 3
    assert optimal_rounding(1.0020000) == 3
    assert optimal_rounding(1.0) == 1
    assert optimal_rounding(0.) == 1


def test_splice():
    assert splice([[1, 2, 3, 4], []]) == [1, 2, 3, 4]
    assert splice([[1, 2, 3], [4]]) == [1, 2, 3, 4]
    assert splice([[], []]) == []
    assert splice([[[[]], []], [[[]]]]) == [[[]], [], [[]]]


def test_is_square():
    assert is_square(0)
    assert is_square(1)
    assert is_square(4)
    assert is_square(1000000000000)
    assert is_square(4.0)
    assert is_square(16.000)
    assert not is_square(-9)
    assert not is_square(3)
    assert not is_square(16.0000000001)
