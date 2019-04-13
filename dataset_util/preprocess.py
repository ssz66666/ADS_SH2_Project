#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

DEFAULT_WINDOW_SIZE = 100
DEFAULT_WINDOW_OVERLAP = 0

def query_to_sliding_windows(cur, size=DEFAULT_WINDOW_SIZE, overlap=DEFAULT_WINDOW_OVERLAP):
    return to_sliding_windows(cur, list(map(lambda x: x[0], cur.description)), size, overlap)

def to_sliding_windows(rows, col_headings=None, size=DEFAULT_WINDOW_SIZE, overlap=DEFAULT_WINDOW_OVERLAP):
    if size <= overlap:
        raise ValueError("size must be strictly greater than overlap")
    count = 0
    arr = []
    for row in rows:
        arr.append(row)
        count += 1
        if count >= size:
            _arr = arr
            count = overlap
            arr = arr[size-overlap:]
            yield pd.DataFrame.from_records(_arr, columns=col_headings)
    return
            
def test():
    x = [(1,"2",3.0), (4,"5",6.0),(7,"8",9.0)]
    slided = list(to_sliding_windows(x, col_headings=None, size=2))
    assert len(slided) == 1
    assert slided[0].equals(pd.DataFrame([[1,"2",3.0], [4,"5",6.0]]))
    
if __name__ == "__main__":
    test()

        
            
            