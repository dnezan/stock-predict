import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
import sklearn.model_selection as sk
import sklearn.preprocessing as pp
from sklearn import metrics
from sklearn.neural_network import MLPRegressor

data_df = pd.read_csv('nasdaq.csv')
data_df.time=pd.to_datetime(data_df['Date'], format='%m-%d-%Y')
y = data_df['Adj_Close']

#Feature Generation
data_df['Adj_Close_S'] = data_df['Adj_Close'].shift(-1)
data_df['DailyReturns'] = ((data_df['Adj_Close']-data_df['Adj_Close_S'])/data_df['Adj_Close'])
data_df['UpDown'] = data_df['DailyReturns'] > 0
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
#del data_df['Adj_Open']

data_df.to_csv('TEMP.csv', sep = ',')
print('FINALIZED TEST DATA')

#FINAL TARGET CSV
path1 = 'TEMP.csv'
path2 = 'finaltarget.csv'
outdata = []
i = 0
inp = open(path1, 'r', newline='')
output = open(path2, 'w', newline='')
reader = csv.reader(inp, delimiter=',')
writer = csv.writer(output, delimiter=',')
for row in reader:
            outdata.append([row[0], row[2], row[3], row[4], row[7], row[6], row[5]])
writer.writerows(outdata)

f = open('finaltarget.csv')
f.readline()  # skip the header
data = np.loadtxt(f, delimiter=',') #,converters={1: strpdate2num('%m-%d-%Y')})
X=data[:,1:6]
y=data[:,6]

#Validation
X_train, X_test, y_train, y_test = sk.train_test_split(X, y, test_size=0.33)
scaler = pp.StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

np.savetxt('xtrain.csv', X_train, delimiter = ',')
np.savetxt('ytrain.csv', y_train, delimiter = ',')
np.savetxt('xtest.csv', X_test, delimiter = ',')
np.savetxt('ytest.csv', y_test, delimiter = ',')

print('NEURAL NET')
#lbfgs solver
mlp = MLPRegressor(solver='lbfgs', hidden_layer_sizes=10, shuffle=True, max_iter=1000, random_state=1, momentum=0.75, alpha=0.4)
mlp.fit(X, y)
z = mlp.predict(X)
np.savetxt('n1.csv', z, delimiter = ',')
np.savetxt('n2.csv', y_test, delimiter = ',')
print("THE MEAN SQUARED ERROR IS", metrics.mean_squared_error(y, z))

#Plotting Adjusted Close Graph
plt.plot(data_df.time[0:1223], y, color='blue', linewidth=0.8, label='Target')
plt.plot(data_df.time[0:1223], z, color='red',  linewidth=0.8, label='Predicted')
plt.ylabel('Adjusted Close')
plt.xlabel('Date')
plt.show()

print("FINISHED")
