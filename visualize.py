import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv

data_df = pd.read_csv('EOD-AAPL.csv')

data_df.time=pd.to_datetime(data_df['Date'], format='%Y-%m-%d')
y = data_df['Adj_Close']

#Feature Generation
data_df['Adj_Close_S'] = data_df['Adj_Close'].shift(-1)

data_df['DailyReturns'] = ((data_df['Adj_Close']-data_df['Adj_Close_S'])/data_df['Adj_Close'])

data_df['UpDown'] = data_df['DailyReturns'] > 0
#print(data_df)
data_df.to_csv('testdata.csv', sep = ',')

#Converting Bool to 0 and 1
path1='testdata.csv'
path2='target.csv'
outdata = []
inp = open(path1,'r', newline='')
output = open(path2,'w', newline='')
reader=csv.reader(inp, delimiter=',')
writer=csv.writer(output,delimiter=',')
for row in reader:
        if row[-1] == 'True':
            row[-1] = 1
        elif row[-1] == 'False':
            row[-1] = 0
        outdata.append([row[-1]])
writer.writerows(outdata)

#Target
target = pd.read_csv('target.csv')
data_df['Target']=target

#del data_df['Adj_Open']
del data_df['Dividend']
del data_df['Split']
del data_df['Adj_Low']
del data_df['Adj_High']
del data_df['Adj_Open']
del data_df['Adj_Volume']
del data_df['Adj_Close_S']

data_df.to_csv('TEMP.csv', sep = ',')

#FINAL TARGET CSV
path1='TEMP.csv'
path2='finaltarget.csv'
outdata = []
i=0
inp = open(path1,'r', newline='')
output = open(path2,'w', newline='')
reader=csv.reader(inp, delimiter=',')
writer=csv.writer(output,delimiter=',')
for row in reader:
        if i <= 8001:
            outdata.append([row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10]])
        i=i+1
writer.writerows(outdata)

#Plotting Adjusted Close Graph
'''
z=data_df['Adj_Close']
plt.interactive(False)
plt.plot(data_df.time,z)
plt.show()
print("FINISHED")
'''