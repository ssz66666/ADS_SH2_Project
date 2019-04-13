from sklearn.ensemble import RandomForestClassifier
import sklearn.model_selection
import pandas as pd
import sqlite3

from dataset_util import uci_mhealth
import pickle

from config import SQLITE_DATABASE_FILE

def main():
    with sqlite3.connect(SQLITE_DATABASE_FILE) as conn:
        mhealth = pd.read_sql_query(uci_mhealth.raw_table_query, conn)
    X, y = uci_mhealth.to_classification(mhealth)
    clsf = RandomForestClassifier(n_estimators=500, class_weight="balanced", n_jobs=-1)
    clsf.fit(X,y)
    # RF_pred = clsf.predict(X_test)
    with open('training_result.pickle', 'wb') as f:
        pickle.dump(clsf, f)

# testset = sklearn.model_selection.train_test_split(dataset, test_size = 0.2, random_state = 1, stratify = dataset.iloc[:,1])
        
if __name__ == "__main__":
    main()
