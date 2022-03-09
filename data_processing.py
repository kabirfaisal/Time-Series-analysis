import time
start_time = time.time()

import csv

#input number you want to search
number = input('Enter specific gas station ID\n')

#read csv, and split on "," the line
with open('......../Downloads/gas_station_information_history_new_data.csv', 'r') as f, open('....../Downloads/data.csv', 'w') as outf:
    reader = csv.reader(f, delimiter=',')
    writer = csv.writer(outf)
    for row in reader:
        if number in row:
            writer.writerow(row)

import pandas as pd

#Read csv file
df = pd.read_csv('......./Downloads/data.csv', header=None, names=["id", "stid", "disel", "e5", "e10", "date", "changed" ] )

#drop duplicates
df = df.drop_duplicates(subset=['date'], keep='first')
df['date'] =  pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')

#set date index
df = df.set_index('date')#.resample('4min').ffill()

#resample every 4 minutes with previous data
df = df.resample('4min').ffill()

#resample every 4 minutes with NaN valoue
#df = df.resample('4min')

#save csv
df.to_csv('...../Downloads/data_P.csv')

print("--- %s seconds ---" % (time.time() - start_time))