from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import sqlite3

from dataset_util import uci_mhealth

with sqlite3.connect('test.db') as conn:
   mhealth = pd.read_sql_query(uci_mhealth.raw_table_query, conn)
x, y = uci_mhealth.to_classification(mhealth)
clsf = RandomForestClassifier(n_estimators=500, class_weight="balanced", n_jobs=-1)
clsf.fit(x,y)
