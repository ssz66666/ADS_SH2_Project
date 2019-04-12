#!/usr/bin/env python
# -*- coding: utf-8 -*-

from enum import Enum

DATASET_NAME = "uci_mhealth"


class Location(Enum):
    CHEST = 1
    LEFT_ANKLE = 2
    RIGHT_LOWER_ARM = 3