#!/usr/bin/env python
# -*- coding: utf-8 -*-

from itertools import product

def mul_str_arr(*arrs, separator='_'):
    return list(map(separator.join, product(*arrs)))
