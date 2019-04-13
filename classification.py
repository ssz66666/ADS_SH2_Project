from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import sqlite3

from dataset_util import uci_mhealth
import pickle

from config import SQLITE_DATABASE_FILE

def main():
    with sqlite3.connect(SQLITE_DATABASE_FILE) as conn:
        mhealth = pd.read_sql_query(uci_mhealth.raw_table_query, conn)
    x, y = uci_mhealth.to_classification(mhealth)
    clsf = RandomForestClassifier(n_estimators=500, class_weight="balanced", n_jobs=-1)
    clsf.fit(x,y)
    with open('training_result.pickle', 'wb') as f:
        pickle.dump(clsf, f)

if __name__ == "__main__":
    main()