import pandas as pd
df = pd.read_csv("f:\\TOR NON-TOR\merged_5s revised Scenario A.csv",sep=',')
attributes = ['SourceIP','Source Port','Destination IP','Destination Port','Protocol','Flow Duration','Flow Bytes/s','Flow Packets/s','Flow IAT Mean','Flow IAT Std','Flow IAT Max','Flow IAT Min','Fwd IAT Mean','Fwd IAT Std','Fwd IAT Max','Fwd IAT Min','Bwd IAT Mean','Bwd IAT Std','Bwd IAT Max','Bwd IAT Min','Active Mean','Active Std','Active Max','Active Min','Idle Mean','Idle Std','Idle Max','Idle Min','label']
df.columns = attributes
print(df.head())
print(df.dtypes)
df['count'] = 1
print(df[['label', 'count']].groupby('label').count())

from sklearn.preprocessing import scale
num_cols1 = ['SourceIP','Source Port','Destination IP','Destination Port','Protocol','Flow Duration','Flow Bytes/s','Flow Packets/s','Flow IAT Mean','Flow IAT Std','Flow IAT Max','Flow IAT Min','Fwd IAT Mean','Fwd IAT Std','Fwd IAT Max','Fwd IAT Min','Bwd IAT Mean','Bwd IAT Std','Bwd IAT Max','Bwd IAT Min','Active Mean','Active Std','Active Max','Active Min','Idle Mean','Idle Std','Idle Max','Idle Min']

df_scaled = (df[num_cols1])

from sklearn.preprocessing import LabelEncoder
df_scaled = pd.DataFrame(df_scaled,columns = num_cols1)

for column in df_scaled.columns:
    if df_scaled[column].dtype == type(object):
        le = LabelEncoder()
        df_scaled[column] = le.fit_transform(df[column])
print(df_scaled.dtypes)



label = ['label']
df_label = pd.DataFrame(df,columns = label)
print(df_scaled.head())

levels = {'nonTOR':0,'TOR':1}
df_scaled['label'] =[levels[x] for x in df['label']]
print(df_scaled.head())

def cond_hist(df,plot_cols,bins = 10):
    import matplotlib.pyplot as plt
    import seaborn as sns
    for col in plot_cols:
        fig = plt.figure()
        ax = fig.gca()
        df[col].plot.hist(ax = ax,bins = bins)
        ax.set_title('Histograms of '+col)
        ax.set_xlabel(col)
        ax.set_ylabel('label')
        plt.show()
cond_hist(df_scaled,attributes)



from sklearn.model_selection import train_test_split
import numpy as np
df_train_features, df_test_features, df_train_label, df_test_label = train_test_split(df_scaled, df_label , test_size=0.2)

df_train_label = np.ravel(df_train_label)
df_test_label = np.ravel(df_test_label)







print(df_train_features.shape)
print(df_train_label.shape)
print(df_test_features.shape)
print(df_test_label.shape)

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import BaggingClassifier
#KNN_mod=svm.SVC()
KNN_mod = KNeighborsClassifier(n_neighbors = 3)
#KNN_mod = (GaussianNB())
KNN_mod.fit(df_train_features, df_train_label)
df_test = pd.DataFrame(df_test_features, columns = num_cols1)
df_test['predicted'] = KNN_mod.predict(df_test_features)
df_test['correct'] = [1 if x == z else 0 for x, z in zip(df_test['predicted'], df_test_label)]
accuracy = 100.0 * float(sum(df_test['correct'])) / float(df_test.shape[0])
print(accuracy)
