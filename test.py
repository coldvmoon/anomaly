import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from mpl_toolkits import mplot3d
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from matplotlib.colors import LogNorm
data = pd.read_json("input.json",orient='records',typ='frame')
selected_data=np.array([data['comUpdate'],data['comInsert']]).T
skf = StratifiedKFold(n_splits=4,shuffle=True,random_state=0)
train_index, test_index = next(iter(skf.split(selected_data,np.ones(600))))
X_train=selected_data[train_index]

estimator = GaussianMixture()
estimator.fit(X_train)

#y_train_pred = estimator.predict(X_train)

#Z = -estimator.score_samples(X_train)

#plt.plot_date(data['dataTime'],data['comInsert'],'-')
#plt.scatter(data['bytesReceived'],data['comInsert']+data['comUpdate'])
#plt.scatter(data['bytesSent'],data['comSelect'])
#plt.hist(((data['comInsert']+data['comUpdate'])/data['bytesReceived']), bins=20)
#density, bins, patches = hist
#widths = bins[1:] - bins[:-1]
#print((density * widths).sum())
#plt.plot(st.poisson.pmf(np.arange(0,1000),108))
#plt.show()


#X,Y=np.split(X_train.T,[1])
x,y=np.split(X_train.T,[1])
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -estimator.score_samples(XX)
Z = Z.reshape(X.shape)

CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),
                 levels=np.logspace(0, 3, 10))
CB = plt.colorbar(CS, shrink=0.8, extend='both')
plt.scatter(X_train[:, 0], X_train[:, 1], .8)

plt.title('Negative log-likelihood predicted by a GMM')
plt.axis('tight')
plt.show()


#newData  = pd.DataFrame({'dataTime':data['dataTime'], 'comInsert':data['comInsert']})
#newData.to_csv("save.csv")


