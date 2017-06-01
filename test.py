import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from mpl_toolkits import mplot3d
data = pd.read_json("input.json",orient='records',typ='frame')
dataTime = data['dataTime']  
#plt.plot_date(data['dataTime'],data['comInsert'],'-')
#plt.scatter(data['bytesReceived'],data['comInsert']+data['comUpdate'])
#plt.scatter(data['bytesSent'],data['comSelect'])
#plt.hist(data['comInsert']**0.0002, bins=200)
#density, bins, patches = hist
#widths = bins[1:] - bins[:-1]
#print((density * widths).sum())
#plt.plot(st.poisson.pmf(np.arange(0,1000),108))
#plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.scatter3D(data['bytesSent'],data['comSelect'], data['comUpdate'], 'gray')

#newData  = pd.DataFrame({'dataTime':data['dataTime'], 'comInsert':data['comInsert']})
#newData.to_csv("save.csv")


