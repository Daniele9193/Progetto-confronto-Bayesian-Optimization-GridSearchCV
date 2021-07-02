import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Normalizer


df = pd.read_csv('repair_log.csv', sep=';')
df.head()

# Elimino le colonne non necessarie
df = df.drop('Variant', axis=1)
df = df.drop('Variant index', axis=1)
df = df.drop('(case) description', axis=1)
df = df.drop('defectFixed', axis=1)
df = df.drop('defectType', axis=1)
df = df.drop('lifecycle:transition', axis=1)
df = df.drop('numberRepairs', axis=1)

df.shape
df.isna().sum()
df.describe(include='all')
df = df.fillna(0)
df.head()
df.isna().sum()

# Trasformo le colonne del timestamp in datetime
df['Start Timestamp'] = pd.to_datetime(df['Start Timestamp'])
df['Complete Timestamp'] = pd.to_datetime(df['Complete Timestamp'])
df.head()

# Creo la colonna timeActivity come differenza del tempo finale e di quello iniziale e la trasformo in minuti
df['timeActivity'] = df['Complete Timestamp'] - df['Start Timestamp']
df['timeActivity'] = df['timeActivity'].apply(lambda x: (x.total_seconds())/60)
df.head(50)

# Raggruppo il dataset per case ID e creo la colonna numberActivity
grouped = df.groupby(['Case ID'], sort=False)
grouped.head()
df_new = pd.DataFrame()
df_new['numberActivity'] = grouped['Activity'].count()
df_new.head()

# Creo la colonna differentResources
df_new['differentResources'] = grouped['Resource'].nunique()
df_new.head()

# Creo colonna minTime come tempo iniziale minimo
df_new['minTime'] = grouped['Start Timestamp'].min()
df_new.head()

# Creo colonna maxTime come tempo iniziale massimo
df_new['maxTime'] = grouped['Complete Timestamp'].max()
df_new.head()

# Trasformo minTime e maxTime in datetime
df_new['minTime'] = pd.to_datetime(df_new['minTime'])
df_new['maxTime'] = pd.to_datetime(df_new['maxTime'])

# Creo colonna timeDiff come differenza tra maxTime e minTime e la converto in minuti
# Creo la colonna avgTimeActivity
df_new.head()
df_new['timeDiff'] = df_new['maxTime'] - df_new['minTime']
df_new['timeDiff'] = df_new['timeDiff'].apply(lambda x: (x.total_seconds())/60)
df_new.head()
df_new.dtypes
df_new['avgTimeActivity'] = df_new.timeDiff/df_new.numberActivity
df_new.head()

# Creo colonna phoneType
df_new['phoneType'] = grouped['phoneType'].unique()
df_new.head()

def phone_type(phones):
    flag = 'NT'
    for p in phones:
        if p != 0:
            flag = p
    return flag

df_new.phoneType = df_new.phoneType.apply(lambda x: phone_type(x))
df_new.head()
a = df_new.groupby(['phoneType']).count()
a.head()
le = LabelEncoder()
le.fit(df_new.phoneType)
le.classes_
df_new.phoneType = le.transform(df_new.phoneType)
df_new.head()

df_new['timeDiff'].describe()

# Creo colonna ratioResourceActivity
df_new['ratioResourceActivity'] = df_new.differentResources / df_new.numberActivity
df_new.head()

# Elimino le colonne minTime e maxTime
df_new = df_new.drop('minTime', axis=1)
df_new = df_new.drop('maxTime', axis=1)

# Inserimento colonne numSystem, numSolver, numTester
df.groupby('Resource').count()
def num_system(resources):
    count = 0
    for res in resources:
        if res in 'System':
            count += 1
    return count

def num_solver(resources):
    count = 0
    for res in resources:
        if 'Solver' in res:
            count += 1
    return count

def num_tester(resources):
    count = 0
    for res in resources:
        if 'Tester' in res:
            count += 1
    return count

df_new['numSystem'] = grouped['Resource'].apply(lambda x: num_system(x))
df_new['numSolver'] = grouped['Resource'].apply(lambda x: num_solver(x))
df_new['numTester'] = grouped['Resource'].apply(lambda x: num_tester(x))
df_new.head()

# Inserimento colonne timeSystem, timeSolver, timeTester
df_new['listResources'] = grouped['Resource'].apply(list)
df_new['listTime'] = grouped['timeActivity'].apply(list)
df_new.head()
def time_system(row):
    count = 0
    for res,time in zip(row['listResources'], row['listTime']):
        if 'System' in res:
            count += time
    return count

def time_solver(row):
    count = 0
    for res,time in zip(row['listResources'], row['listTime']):
        if 'Solver' in res:
            count += time
    return count

def time_tester(row):
    count = 0
    for res,time in zip(row['listResources'], row['listTime']):
        if 'Tester' in res:
            count += time
    return count

df_new['timeSystem'] = df_new.apply(time_system, axis=1)
df_new['timeSolver'] = df_new.apply(time_solver, axis=1)
df_new['timeTester'] = df_new.apply(time_tester, axis=1)
df_new.head()

df_new = df_new.drop('listResources', axis=1)
df_new = df_new.drop('listTime', axis=1)

df_new.shape
df_new.describe()

df_new.head()


# Salvataggio del DataFrame in un file csv
df_new.to_csv ('dataset.csv', index = True, header=True)


# Normalizzazione e OneHotEncoding
df = pd.read_csv('dataset.csv', sep=',')
df.head()
df.isna().sum()
df = df.drop('timeSystem', axis=1)  # Tutti valori 0
df = df.drop('Case ID', axis=1)
df.columns

# One Hot Encoder phoneType column
enc = OneHotEncoder(handle_unknown='ignore')
enc_df = pd.DataFrame(enc.fit_transform(df[['phoneType']]).toarray())
df = df.join(enc_df)
df = df.drop('phoneType', axis=1)
df = df.rename(columns={0: "phoneType0", 1: "phoneType1", 2: "phoneType2"})
df.head()
df.isna().sum()

# Normalization continuos features
x = df[['numberActivity', 'differentResources','timeDiff','avgTimeActivity', 'numSystem', 'numSolver', 'numTester','timeSolver','timeTester']] #returns a numpy array
transformer = Normalizer().fit(x)
t = transformer.transform(x)
df_normalized = pd.DataFrame(t)
df = df.join(df_normalized)
df = df.drop(['numberActivity', 'differentResources','timeDiff','avgTimeActivity', 'numSystem', 'numSolver', 'numTester','timeSolver','timeTester'], axis=1)
df = df.rename(columns={0: "numberActivity", 1: "differentResources", 2: "timeDiff", 3: "avgTimeActivity", 4: "numSystem", 5: "numSolver", 6: "numTester", 7: "timeSolver",8: "timeTester"})
df.head()

df.to_csv('dataset_normalized.csv', index = False, header=True)
