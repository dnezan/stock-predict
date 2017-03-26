import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
import sklearn.model_selection as sk
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


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
        if i <= 8001:
            outdata.append([row[0], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[10]])
        i=i+1
writer.writerows(outdata)

#y=np.genfromtxt('target.csv', delimiter=',')
f = open('finaltarget.csv')
f.readline()  # skip the header
data = np.loadtxt(f, delimiter=',') #,converters={1: strpdate2num('%m-%d-%Y')})

X=data[:,1:7]
y=data[:,8]

#Validation
X_train, X_test, y_train, y_test = sk.train_test_split(X, y, test_size=0.33, random_state=42)
#print(X_train)

np.savetxt('q.csv', X_train, delimiter = ',')
np.savetxt('r.csv', y_train, delimiter = ',')

#clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
#print(clf.score(X.test, y.test))

print('RUNNING ALGORITHM')

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100, 20), random_state=1)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

'''
count=1
while count < 50:
    knn = KNeighborsClassifier(n_neighbors=count)
    knn.fit(X_train,y_train)
    print(knn.score(X_test, y_test))
    count = count + 1
'''

print('FINISHED')

#print(X)
'''
#Plotting Adjusted Close Graph
z=data_df['Adj_Close']
plt.interactive(False)
plt.plot(y_train,y_test)
plt.show()
print("FINISHED")

'''
