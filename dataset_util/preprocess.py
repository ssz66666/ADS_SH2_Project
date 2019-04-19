#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

DEFAULT_WINDOW_SIZE = 100
DEFAULT_WINDOW_OVERLAP = 0

def query_to_sliding_windows(cur, *args, **kwargs):
    return to_sliding_windows_cursor(cur, list(map(lambda x: x[0], cur.description)), *args, **kwargs)

def to_sliding_windows_cursor(cur, col_headings=None, size=DEFAULT_WINDOW_SIZE, overlap=DEFAULT_WINDOW_OVERLAP):
    if size <= overlap:
        raise ValueError("size must be strictly greater than overlap")
    l_arr = 0
    arr = []
    while True:
        fetched = cur.fetchmany(size - l_arr)
        if (l_arr + len(fetched)) < size:
            return
        else:
            arr = arr + fetched
            _arr = np.vstack(arr)
            l_arr = overlap
            arr = arr[size-overlap:]
            yield pd.DataFrame(_arr, columns=col_headings)
    return

# returns X, y, subject_id
def to_classification(df):
    return df.iloc[:,2:], df.iloc[:,0], df.iloc[:,1]

def to_sliding_windows(rows, col_headings=None, size=DEFAULT_WINDOW_SIZE, overlap=DEFAULT_WINDOW_OVERLAP):
    if size <= overlap:
        raise ValueError("size must be strictly greater than overlap")
    count = 0
    arr = []
    for row in rows:
        arr.append(row)
        count += 1
        if count >= size:
            _arr = np.vstack(arr)
            count = overlap
            arr = arr[size-overlap:]
            yield pd.DataFrame(_arr, columns=col_headings)
    return
            
def test():
    x = [(1,"2",3.0), (4,"5",6.0),(7,"8",9.0)]
    slided = list(to_sliding_windows(x, col_headings=None, size=2))
    assert len(slided) == 1
    assert slided[0].equals(pd.DataFrame([[1,"2",3.0], [4,"5",6.0]]))
    
if __name__ == "__main__":
    test()

        
            
            