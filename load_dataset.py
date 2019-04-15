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
                conn.commit()

UPDATE_PAMAP2 = False
NEW_PAMAP2_PATH = "raw_dataset/uci_pamap2/PAMAP2_finalDataset"

def main():
    with sqlite3.connect(SQLITE_DATABASE_FILE) as conn:
        load_all_datasets(conn, RAW_DATASET_DIR, *DATASET_LIST)
        if UPDATE_PAMAP2:
            m = get_dataset_loader_module("uci_pamap2")
            m._HACK_force_store_updated_dataset_to_sql(conn.cursor(), NEW_PAMAP2_PATH)

if __name__ == "__main__":
    main()