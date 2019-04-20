from cluster_comparison import cv_main
import pandas as pd


def label_mapping(df_mapping):#,dataset_features):

    mapping = []
    maps = []
    for i in range(0,6):
        max_index = df_mapping.iloc[:,i].idxmax(axis = 0) +1 # mhealth
        maps.append(i+1) # pamap
        maps.append(max_index)

        mapping.append(maps)
        maps = []

    maps = [5,10]
    mapping.append(maps)

    for i in mapping:
        print(i)
    return mapping

#def pamap_label_mapping(df_mapping,pamap_dataset):

    #for i in range(7,)
    #subject_id = pamap_dataset.loc[pamap_dataset['activity_id'] == i]

if __name__ == "__main__":
    df_mapping = cv_main()

    mapping = label_mapping(df_mapping)
    #print(df_mapping)
    #exit(0)
    pamap_dataset = pd.read_pickle("pamap_features.pkl")




    pd.read_pickle("mhealth_features.pkl")



