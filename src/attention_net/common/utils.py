#!/usr/bin/env python2

import datetime
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

def configure_loggers():
    LOGGER.setLevel(SUBDEBUG)
    debug_handler = logging.StreamHandler()
    debug_handler.setLevel(INFO)
    debug_formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%m-%d %H:%M')
    debug_handler.setFormatter(debug_formatter)
    LOGGER.addHandler(debug_handler)
    logfile = LOG_DIR + "/" + datetime.datetime.now().strftime("%Y.%m.%d-%H.%M.%S.%f") + ".log"
    ensure_dir(logfile)
    file_handler = logging.FileHandler(logfile)
    file_handler.setLevel(SUBDEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    LOGGER.addHandler(file_handler)
    # Necessary to circumvent the non-configurable logger of pyNN
    logging.root.addHandler(file_handler)
       

def log_tick(s):
    now = datetime.datetime.now()
    LOGGER.log(SUBDEBUG, "tick at time %s: %s", now, s)
    for h in LOGGER.handlers:
       h.flush()
