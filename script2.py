import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, MeanShift
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from statistics import mean

df = pd.read_csv('dataset_normalized.csv', sep=',')
df.head()

################################# K-MEANS ######################################
df_kmean = pd.DataFrame()
silhouette = []
dabo = []
K = range(2,11)

for k in K:
    kmeanModel = KMeans(n_clusters=k, random_state=42).fit(df)
    labels = kmeanModel.labels_
    labelset = set(labels)
    if len(labelset) > 1:
        db, db_std = davies_bouldin_score(df, labels)
        sil, sil_std = silhouette_score(df, labels)
        new_row = {'k':k, 'davies_bouldin_mean':db, 'davies_bouldin_std': db_std, 'silhouette_mean':sil, 'silhouette_std':sil_std, 'clusters':len(labelset)}
        df_kmean = df_kmean.append(new_row, ignore_index=True)
    else:
        new_row = {'k':k, 'davies_bouldin_mean':1, 'davies_bouldin_std': 0, 'silhouette_mean':-1, 'silhouette_std':0, 'clusters':1}
        df_kmean = df_kmean.append(new_row, ignore_index=True)


df_kmean.head(15)
df_kmean.to_csv('kmean_tot.csv')



################################# DBSCAN #######################################
df_dbscan = pd.DataFrame()
eps_var = np.linspace(0.02,1,50)
min_sampl = range(2,11)
for e in eps_var:
    silhouette = []
    dabo = []
    for m_sampl in min_sampl:
        dbscan = DBSCAN(eps=e, min_samples=m_sampl)
        dbscan.fit(df)
        labels = dbscan.labels_
        labelset = set(labels)
        if len(labelset) > 1:
            sil, sil_std = silhouette_score(df,labels)
            db, db_std = davies_bouldin_score(df, labels)
            new_row = {'eps':e, 'min_samples':m_sampl, 'davies_bouldin_mean':db, 'davies_bouldin_std': db_std, 'silhouette_mean':sil, 'silhouette_std':sil_std, 'clusters':len(labelset)}
            df_dbscan = df_dbscan.append(new_row, ignore_index=True)
        else:
            new_row = {'eps':e, 'min_samples':m_sampl, 'davies_bouldin_mean':1, 'davies_bouldin_std':0, 'silhouette_mean':-1, 'silhouette_std':0}
            df_dbscan = df_dbscan.append(new_row, ignore_index=True)

df_dbscan.head(50)
df_dbscan['silhouette_mean'].max()
df_dbscan['davies_bouldin_mean'].min()
df_dbscan.to_csv('dbscan_tot.csv')





################################# MEAN SHIFT #######################################
df_meanshift = pd.DataFrame()
bandwidth_var = np.linspace(0.02,1,50)
for b in bandwidth_var:
    meanshiftModel = MeanShift(bandwidth=b).fit(df)
    labels = meanshiftModel.labels_
    labelset = set(labels)
    if len(labelset) > 1:
        db, db_std = davies_bouldin_score(df, labels)
        sil, sil_std = silhouette_score(df, labels)
        new_row = {'bandwidth':b, 'davies_bouldin_mean':db, 'davies_bouldin_std': db_std, 'silhouette_mean':sil, 'silhouette_std':sil_std, 'clusters':len(labelset)}
        df_meanshift = df_meanshift.append(new_row, ignore_index=True)
    else:
        silhouette.append(-1)
        dabo.append(1)
        new_row = {'bandwidth':b, 'davies_bouldin_mean':1, 'davies_bouldin_std': 0, 'silhouette_mean':-1, 'silhouette_std':0, 'clusters':1}
        df_meanshift= df_meanshift.append(new_row, ignore_index=True)

df_meanshift.head(50)


df_meanshift['davies_bouldin_mean'] = df_meanshift['davies_bouldin_mean'].apply(lambda x: -x)
df_meanshift.to_csv('meanshift_tot.csv')
