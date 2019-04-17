from sklearn.ensemble import RandomForestClassifier
import sklearn.model_selection
import pandas as pd
import sqlite3

from datetime import datetime as dt

from dataset_util import uci_mhealth
from dataset_util.extract_input_features import all_feature, extract_features
import matplotlib.pyplot as plt

import numpy as np
from config import SQLITE_DATABASE_FILE, TRAINING_SET_PROPORTION
from scikitplot.metrics import plot_confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns




def main():
    with sqlite3.connect(SQLITE_DATABASE_FILE) as conn:
        # features = pd.read_sql_query(uci_mhealth.raw_table_valid_data_query, conn)
        # sliding_windows = uci_mhealth.to_sliding_windows(conn)
        # subject_ids = uci_mhealth.get_subject_ids(conn)
        activity_ids = uci_mhealth.get_activity_ids(conn)

        data = conn.execute(uci_mhealth.raw_table_query)
        # data = data.fetchall()
    print(activity_ids)
    data_x = []; data_y = []; data_z = []; label = []
    for row in tqdm(data):
        if row[0] != 0:
            label.append(row[0])
            data_x.append(row[3])
            data_y.append(row[4])
            data_z.append(row[5])

    colours = ['b', 'r', 'g', 'k', 'm', 'c', 'y', 'slategray', 'plum', 'teal', 'coral', 'gold']
    labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']

    colour = []
    # print(label)
    for i in tqdm(range(len(label))):
        c = colours[label[i]-1]
        # print(label[i])
        # print(c)
        colour.append(c)
    # ax = plt.axes(projection='3d')
    # ax.scatter3D(data_x, data_y, data_z, c=colour, marker='o', s=50)

    # lb1 = ax.scatter(1.5, 1, c='b', label='1'); lb2 = ax.scatter(0.3, 0.3, c='r', label='2'); lb3 = ax.scatter(0.7, 0.9, c='g', label='3') ;lb4 = ax.scatter(1.2, 0.2, c='k', label='4')
    # lb5 = ax.scatter(1.5, 1, c='m', label='5'); lb6 = ax.scatter(0.3, 0.3, c='c', label='6'); lb7 = ax.scatter(0.7, 0.9, c='y', label='7') ;lb8 = ax.scatter(1.2, 0.2, c='slategray', label='8')
    # lb9 = ax.scatter(1.5, 1, c='plum', label='9'); lb10 = ax.scatter(0.3, 0.3, c='teal', label='10'); lb11 = ax.scatter(0.7, 0.9, c='coral', label='11') ;lb12 = ax.scatter(1.2, 0.2, c='gold', label='12')
    #
    #
    # plt.legend(handles=[lb1, lb2, lb3, lb4, lb5, lb6, lb7, lb8, lb9, lb10, lb11, lb12], loc=1, title=r'$\bf{Activity}$')
    # lb1.remove(); lb2.remove(); lb3.remove(); lb4.remove(); lb5.remove(); lb6.remove()
    # lb7.remove(); lb8.remove(); lb9.remove(); lb10.remove(); lb11.remove(); lb12.remove()

    # plt.legend(handles=labels, loc=1, title=r'$\bf{Activity}$')

    # plt.legend(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'])
    # plt.savefig('mhealth_3D_visual.png')
    # plt.clf()
    # plt.show()

    pca = PCA(n_components=2)
    df = pd.DataFrame()
    df['X'] = data_x; df['Y'] = data_y; df['Z'] = data_z;
    x = StandardScaler().fit_transform(df)
    # ax = sns.scatterplot(x="feature 1", y="feature 2", hue='type', data=df)

    pc = (pca.fit_transform(x))

    df['f1'] = pc[:, 0]; df['f2'] = pc[:, 1];  df['Label'] = label
    plt.scatter(pc[:, 0], pc[:, 1], c=colour, marker='o')

    lb1 = plt.scatter(1.5, 1, c='b', label='1'); lb2 = plt.scatter(0.3, 0.3, c='r', label='2'); lb3 = plt.scatter(0.7, 0.9, c='g', label='3') ;lb4 = plt.scatter(1.2, 0.2, c='k', label='4')
    lb5 = plt.scatter(1.5, 1, c='m', label='5'); lb6 = plt.scatter(0.3, 0.3, c='c', label='6'); lb7 = plt.scatter(0.7, 0.9, c='y', label='7') ;lb8 = plt.scatter(1.2, 0.2, c='slategray', label='8')
    lb9 = plt.scatter(1.5, 1, c='plum', label='9'); lb10 = plt.scatter(0.3, 0.3, c='teal', label='10'); lb11 = plt.scatter(0.7, 0.9, c='coral', label='11') ;lb12 = plt.scatter(1.2, 0.2, c='gold', label='12')


    plt.legend(handles=[lb1, lb2, lb3, lb4, lb5, lb6, lb7, lb8, lb9, lb10, lb11, lb12], loc=1, title=r'$\bf{Activity}$')
    lb1.remove(); lb2.remove(); lb3.remove(); lb4.remove(); lb5.remove(); lb6.remove()
    lb7.remove(); lb8.remove(); lb9.remove(); lb10.remove(); lb11.remove(); lb12.remove()

    # plt.legend(labels)

    # ax = sns.scatterplot(x="f1", y="f2", hue='Label', data=df)
    plt.show()



if __name__ == "__main__":
    main()

