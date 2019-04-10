#!/usr/bin/env python
# -*- coding: utf-8 -*-

import importlib
from importlib.util import find_spec

# sql loader:
# a dataset should only use tables with name prefixed by the '{dataset_name}__'
# e.g. "uci_smartphone.data" is a table for dataset "uci_smartphone"

DATASET_SQLITE_LOADER_FUNC = "import_dataset_to_sqlite"    

def get_dataset_loader_module(name):
    mod = locals().get(name, None)
    if mod is None:
        if find_spec(".{}".format(name), __package__):
            importlib.import_module(name, __package__)
            mod = locals().get(name, None)
    return mod

def get_dataset_loader(name, loader_func_name):
    mod = get_dataset_loader_module(name)
    if mod is not None:
        return getattr(mod, loader_func_name, None)

def get_dataset_sqlite_loader(name):
    get_dataset_loader(name, DATASET_SQLITE_LOADER_FUNC)
        
