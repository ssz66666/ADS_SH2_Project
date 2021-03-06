import dataset_loader.uci_mhealth as mhealth
import sqlite3
import csv

with sqlite3.connect('test.db') as conn:
    cur = conn.cursor()
    # path = 'raw_dataset/uci_pamap2'
    path = 'raw_dataset/uci_mhealth'
    # data = mhealth.load_dataset_to_mem(path)
    mhealth.load_dataset_to_sqlite(conn.cursor(), path)

    # sql_samples = conn.execute("SELECT * FROM {}".format(mhealth.samples_table))
    # sql_sensor = conn.execute("SELECT * FROM {}".format(mhealth.sensor_readings_table))

    conn.commit()

    # with open('samples.csv', 'w') as f1:
    #     writer = csv.writer(f1)
    #     writer.writerow(mhealth.samples_schema.keys())
    #     writer.writerows(sql_samples)


    # with open('sensors.csv', 'w') as f2:
    #     writer2 = csv.writer(f2)
    #     writer2.writerow(list(mhealth.sensor_readings_schema.keys())[:-2])
    #     writer2.writerows(sql_sensor)


