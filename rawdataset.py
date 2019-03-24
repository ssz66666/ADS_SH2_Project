import json
import urllib.request as req
import urllib.parse as urlp
import shutil
import pathlib
import os

from itertools import chain

import asyncio
import concurrent.futures

def download_http_res(url, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    filename = urlp.unquote(urlp.urlsplit(url).path.split('/')[-1])
    with req.urlopen(url) as rsp:
        cd_hdr = rsp.getheader("Content-Disposition", None)
        if cd_hdr != None:
            #try to find the file name from response header
            xs = cd_hdr.split("filename*=UTF-8\'\'")
            if len(xs) == 2:
                filename = urlp.unquote(xs[1].strip('\"'))
            else:
                ys = cd_hdr.split("filename=")
                if len(ys) == 2:
                    filename = urlp.unquote(ys[1].strip('\"'))
        fpath = pathlib.Path(dest_dir, filename)
        with fpath.open('wb') as out_f:
            shutil.copyfileobj(rsp, out_f)

def _make_pair(x):
    return map(lambda u: (x, u), x["data"])

def download_rscs(rscs, dest_dir="raw_dataset"):
    flatten = chain.from_iterable([_make_pair(x) for x in rscs])
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        ftr_file_map = {executor.submit(download_http_res, pair[1], pathlib.Path(dest_dir, pair[0]["name"])): pair[1] for pair in flatten}
        for future in concurrent.futures.as_completed(ftr_file_map):
            name = ftr_file_map[future]
        try:
            future.result()
        except Exception as exc:
            print('failed to download %r with exception: %s' % (name, exc))

if __name__ == "__main__":
    with open("datasets.json", 'r') as dsets:
        download_rscs(json.load(dsets).values())