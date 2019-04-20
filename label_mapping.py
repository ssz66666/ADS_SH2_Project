from cluster_comparison import cv_main

df_mapping = cv_main()

print(df_mapping)

mapping = []
maps = []
for i in range(0,6):
    max_index = df_mapping.iloc[:,i].idxmax(axis = 0) +1
    maps.append(i) # pamap
    maps.append

    print(max_index)



