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

import datetime
import itertools
import logging
import multiprocessing
import os
import sys
import time


from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL, FATAL
from multiprocessing import SUBDEBUG, SUBWARNING
LOGGER = multiprocessing.get_logger()
LOG_DIR = "./logs"

def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

def make_logfile_name():
    return LOG_DIR + "/" + datetime.datetime.now().strftime("%Y.%m.%d-%H.%M.%S.%f") + ".log"

def configure_loggers(debug_handler, file_handler, logger=LOGGER):
    logger.setLevel(SUBDEBUG)
    debug_formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%m-%d %H:%M')
    debug_handler.setFormatter(debug_formatter)
    logger.addHandler(debug_handler)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    # Necessary to circumvent the non-configurable logger of pyNN
    logging.root.addHandler(file_handler)
       

def log_tick(s, logger=LOGGER):
    now = datetime.datetime.now()
    logger.log(SUBDEBUG, "tick at time %s: %s", now, s)
    for h in logger.handlers:
       h.flush()

def optimal_rounding(timestep):
    return len(str(timestep).split('.')[1])

def splice(deep_list):
    return list(itertools.chain.from_iterable(deep_list))

def is_square(n):
    try:
        if n == 0 or n == 1:
            return True
        x = n // 2
        seen = set([x])
        while x * x != n:
            x = (x + (n // x)) // 2
            if x in seen:
                 return False
            seen.add(x)
        return True
    except ZeroDivisionError:
        return False
