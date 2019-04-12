#!/usr/bin/env python
# -*- coding: utf-8 -*-

from enum import Enum
from dataset_loader.uci_pamap2 import samples_table, sensor_readings_table

DATASET_NAME = "uci_pamap2"

class Location(Enum):
    CHEST = 1
    ANKLE = 2
    HAND = 3

