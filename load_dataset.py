#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataset_loader.util import get_dataset_loader_module
import sqlite3
import os

from config import SQLITE_DATABASE_FILE, RAW_DATASET_DIR

DATASET_LIST = [
    "uci_mhealth",
    "uci_pamap2",
]

def load_all_datasets(conn, path, *datasets):
    cur = conn.cursor()
    for dataset in datasets:
        raw_path = os.path.join(path, dataset)
        m = get_dataset_loader_module(dataset)
        if m is not None:
            if m.check_sqlite_table_not_exists(cur):
                m.load_dataset_to_sqlite(cur, raw_path)

def main():
    with sqlite3.connect(SQLITE_DATABASE_FILE) as conn:
        load_all_datasets(conn, RAW_DATASET_DIR, *DATASET_LIST)

if __name__ == "__main__":
    main()