import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, MeanShift
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from statistics import mean
from bayes_opt import BayesianOptimization


df = pd.read_csv('dataset_normalized.csv', sep=',')
df.head()


# Bayesian Optimization KMeans #################################################
df_kmeans_bo = pd.DataFrame()
parameter = []
silhouette_mean = []
silhouette_std = []
davies_bouldin_mean = []
davies_bouldin_std = []
clusters = []

pbounds = {'x': (2,10)}
def kmeans_optimization(x):
    #num_clusters = int(x)
    #parameter.append(num_clusters)
    parameter.append(int(x))
    kmean = KMeans(n_clusters=int(x)).fit(df)
    db, db_std = davies_bouldin_score(df,kmean.labels_)
    sil, sil_std = silhouette_score(df,kmean.labels_)
    clusters.append(len(set(kmean.labels_)))
    silhouette_mean.append(sil)
    silhouette_std.append(sil_std)
    davies_bouldin_mean.append(db)
    davies_bouldin_std.append(db_std)
    return -db


optimizer = BayesianOptimization(
    f=kmeans_optimization,
    pbounds=pbounds,
    random_state=1,
    verbose=2,
)

optimizer.maximize(
    init_points=3,
    n_iter=2,
)

print(optimizer.max)


print(parameter)
print(silhouette_mean)
print(silhouette_std)
print(davies_bouldin_mean)
print(davies_bouldin_std)


df_kmeans_bo = pd.DataFrame({'k':parameter, 'davies_bouldin_mean':davies_bouldin_mean, 'davies_bouldin_std':davies_bouldin_std, 'silhouette_mean':silhouette_mean, 'silhouette_std':silhouette_std, 'clusters':clusters})
df_kmeans_bo.head()

df_kmeans_bo.to_csv('kmeans_bo.csv')







# Bayesian Optimization DBSCAN #################################################
df_dbscan_bo = pd.DataFrame()
parameter_x = []
parameter_y = []
silhouette_mean = []
silhouette_std = []
davies_bouldin_mean = []
davies_bouldin_std = []
clusters = []

pbounds = {'x': (0.02,1), 'y': (2,10)}
def dbscan_optimization(x, y):
    min_sampl = int(y)
    parameter_x.append(x)
    parameter_y.append(min_sampl)
    dbscan = DBSCAN(eps=x, min_samples=min_sampl).fit(df)
    clusters.append(len(set(dbscan.labels_)))
    db, db_std = davies_bouldin_score(df,dbscan.labels_)
    sil, sil_std = silhouette_score(df,dbscan.labels_)
    silhouette_mean.append(sil)
    silhouette_std.append(sil_std)
    davies_bouldin_mean.append(db)
    davies_bouldin_std.append(db_std)
    return sil

optimizer = BayesianOptimization(
    f=dbscan_optimization,
    pbounds=pbounds,
    random_state=1,
    verbose=2,
)

optimizer.maximize(
    init_points=5,
    n_iter=10,
)



print(optimizer.max)

print(parameter_x)
print(parameter_y)
print(silhouette_mean)
print(silhouette_std)
print(davies_bouldin_mean)
print(davies_bouldin_std)


df_dbscan_bo = pd.DataFrame({'eps':parameter_x, 'min_pts':parameter_y, 'davies_bouldin_mean':davies_bouldin_mean, 'davies_bouldin_std':davies_bouldin_std, 'silhouette_mean':silhouette_mean, 'silhouette_std':silhouette_std, 'clusters':clusters})
df_dbscan_bo.head()
df_dbscan_bo.to_csv('dbscan_bo.csv')






# Bayesian Optimization MeanShift #################################################
df_meanshift_bo = pd.DataFrame()
parameter = []
silhouette_mean = []
silhouette_std = []
davies_bouldin_mean = []
davies_bouldin_std = []
clusters = []

pbounds = {'x': (0.02,1)}
def meanshift_optimization(x):
    parameter.append(x)
    meanshift = MeanShift(bandwidth=x).fit(df)
    db, db_std = davies_bouldin_score(df,meanshift.labels_)
    sil, sil_std = silhouette_score(df,meanshift.labels_)
    clusters.append(len(set(meanshift.labels_)))
    silhouette_mean.append(sil)
    silhouette_std.append(sil_std)
    davies_bouldin_mean.append(db)
    davies_bouldin_std.append(db_std)
    return -db


optimizer = BayesianOptimization(
    f=meanshift_optimization,
    pbounds=pbounds,
    random_state=1,
    verbose=2,
)

optimizer.maximize(
    init_points=5,
    n_iter=10,
)

print(optimizer.max)

print(parameter)
print(silhouette_mean)
print(silhouette_std)
print(davies_bouldin_mean)
print(davies_bouldin_std)


df_meanshift_bo = pd.DataFrame({'bandwidth':parameter, 'davies_bouldin_mean':davies_bouldin_mean, 'davies_bouldin_std':davies_bouldin_std, 'silhouette_mean':silhouette_mean, 'silhouette_std':silhouette_std, 'clusters':clusters})
df_meanshift_bo.head()
df_meanshift_bo.to_csv('meanshift_bo.csv')
