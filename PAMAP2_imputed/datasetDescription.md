# PAPMAP2 Dataset

## Changes:

* add subjects information
* add column name
* remove invalid sensor data (i.e. acceleration data of ±6g scale and orientation)
* impute missing values
* remove rows with activities = 0

## Download: 
https://uob-my.sharepoint.com/:u:/g/personal/ec18242_bristol_ac_uk/Eacfw20iwptNkFamV8Uauq0BMBr4v5ut-dVO7rNm3LZByw?e=Qo41gH

## Import to database (temporary):

1. Download the `PAMAP2_finalDataset.zip` and put it into `raw_dataset/uci_pamap2/` directory.
2. Unzip it so that you get a folder `raw_dataset/uci_pamap2/PAMAP2_finalDataset/`, check if this folder contains 8 csv files.
3. Finally, update PAMAP2 dataset with the imputed version by running
```
    python3 load_dataset.py --update-pamap2
```

