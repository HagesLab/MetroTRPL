# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 18:11:39 2023

@author: cfai2
"""
import os
import logging
from datetime import datetime

def start_logging(log_dir="Logs", name=""):

    if not os.path.isdir(log_dir):
        try:
            os.makedirs(log_dir, exist_ok=True)
        except FileExistsError:
            pass

    tstamp = str(datetime.now()).replace(":", "-")
    #logging.basicConfig(filename=os.path.join(log_dir, f"{name}-{tstamp}.log"), filemode='a', level=logging.DEBUG)
    logger = logging.getLogger("Metro Logger Main")
    logger.setLevel(logging.DEBUG)

    handler = logging.FileHandler(os.path.join(log_dir, f"{name}-{tstamp}.log"))
    handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
            fmt='%(asctime)s %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
            )

    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger, handler

def stop_logging(logger, handler, err=0):
    if err:
        logger.error(f"Termining with error code {err}")

    # Spyder needs explicit handler handling for some reason
    logger.removeHandler(handler)
    logging.shutdown()
    return
