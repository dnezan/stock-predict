import matplotlib.pyplot as plt
import pandas as pd
data_df = pd.read_csv('prices.csv')

data_df.time=pd.to_datetime(data_df['date'], format='%m-%d-%Y')
y = data_df['open']

plt.interactive(False)
plt.plot(data_df.time,y)
plt.show()
