#coding=utf-8
import logging

def getLogger():
    logger = logging
    logger.basicConfig(level=logging.INFO)
    return logger