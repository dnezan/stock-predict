import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
import sklearn.model_selection as sk
import sklearn.preprocessing as pp
from sklearn import metrics
from sklearn.neural_network import MLPRegressor
from sklearn import svm
from sklearn import linear_model
from sklearn import utils


import os

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

print('NEURAL NET')
#lbfgs solver
mlp = MLPRegressor(solver='lbfgs', hidden_layer_sizes=10, shuffle=True, max_iter=150, random_state=1, momentum=0.75, alpha=0.4)
mlp.fit(X, y)
z = mlp.predict(X)
print("THE MEAN SQUARED ERROR FOR NEURAL NETWORKS IS", metrics.mean_squared_error(y, z))


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X)
X_norm = sc.transform(X)

clf = svm.SVR(verbose=False, C = 1000.0, epsilon= 0.1, cache_size=10000)
clf.fit(X_norm, y)
z_s = clf.predict(X_norm)
print("THE MEAN SQUARED ERROR FOR SVR IS", metrics.mean_squared_error(z_s, y))

#Plotting Output

f, axarr = plt.subplots(2,2)
axarr[0, 0].set_title('Stock Market Price Prediction')
axarr[0, 0].plot(data_df.time[0:1223], y, color='blue', linewidth=0.8, label='Target')
axarr[0, 0].plot(data_df.time[0:1223], z, color='red',  linewidth=0.8, label='Predicted')
axarr[0, 0].plot(data_df.time[0:1223], z_s, color='green',  linewidth=0.8, label='SVM')

axarr[0, 0].set_xlabel('Date')
axarr[0, 0].set_ylabel('Adjusted Close Price')
legend = axarr[0,0].legend()
frame = legend.get_frame()
frame.set_facecolor('0.90')

#Plotting Errors
i=1
while i < 1223:
    error = z_s-y
    i=i+1
axarr[0,1].set_title('Absolute Error')
axarr[0,1].plot(range(1223), error, color='blue',  linewidth=0.8, label='Error')
axarr[0,1].set_xlabel('Data Point')
axarr[0,1].set_ylabel('Absolute Error')

#Plotting Efficient Epoch
print('CALCULATING EFFICIENT EPOCH')
iter = 50
iter_x = iter
e = []
while iter < 400:
    mlp = MLPRegressor(solver='lbfgs', hidden_layer_sizes=10, shuffle=True, max_iter=iter, random_state=1, momentum=0.75, alpha=0.4)
    mlp.fit(X, y)
    z = mlp.predict(X)
    err=metrics.mean_squared_error(y, z)
    e.append(err)
    iter=iter+1
    print('MAXIMUM ITERATIONS:', iter,', MEAN SQUARE ERROR: ', err)

axarr[1,0].set_title('Epoch Selection')
axarr[1,0].plot(range(iter_x, iter), e, color='blue',  linewidth=0.8, label='Error')
axarr[1,0].set_xlabel('Number of iterations')
axarr[1,0].set_ylabel('Mean Square Error')

plt.show()
print("FINISHED")



