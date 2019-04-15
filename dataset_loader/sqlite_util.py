#!/usr/bin/env python
# -*- coding: utf-8 -*-

# utility functions for sqlite integration

# schema is a dict with (string key: string value) items
# key is the column name, value is the column type
# user is reponsible for making sure the input is valid

RAW_DATASETS_TABLE = "raw_datasets"

# def add_dataset_registry(cur, name):
#     cur.execute("SELECT ")


def build_schema(schema):
    vals = [ v for p in schema.items() for v in p ]
    schema_str = '(' + ('{} {},' * ((len(vals) - 1) // 2)) + '{} {})' 
    return schema_str.format(*vals)

def drop_table_if_exists(cur, table_name):
    cur.execute("DROP TABLE IF EXISTS {};".format(table_name))

def create_table_if_not_exists(cur, table_name, schema):
    create_table_if_not_exists_raw_schema(cur, table_name, build_schema(schema))

def create_table_if_not_exists_raw_schema(cur, table_name, schema_string):
    cur.execute("CREATE TABLE IF NOT EXISTS {} {};".format(table_name, schema_string))

def check_sql_table_exists(cur, table_name):
    return cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table_name,)).fetchone() is not None