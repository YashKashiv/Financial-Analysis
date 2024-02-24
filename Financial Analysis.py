import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sklearn.metrics as metrics
import seaborn as sns

default=pd.read_csv('desktop/Default.csv')

default.head()

# Keeping only required columns

default=default[['default','student','balance','income']]

default.head()

default.shape

default.describe()

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.boxplot(y=default['balance'])

plt.subplot(1,2,2)
sns.boxplot(y=default['income'])
plt.show()

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.countplot(default['student'])

plt.subplot(1,2,2)
sns.countplot(default['default'])
plt.show()

# to see value count in percentage

default['default'].value_counts(normalize=True)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.boxplot(default['default'],default['balance'])

plt.subplot(1,2,2)
sns.boxplot(default['default'],default['income'])
plt.show()

# use normalize='index' and round(2) method for percentage in 2 values

pd.crosstab(default['student'],default['default'],normalize='index').round(2)

# use .corr method to see corelation b/w two data columns
# use annot=True method to see percentage of corelation

sns.heatmap(default[['balance','income']].corr(),annot=True)
plt.show()

# use this method to see null value in data

default.isnull().sum()

# use .quantile([]) method to take certain percentage from data

q1,q3=default['balance'].quantile([.25,.75])

# for interquartile range
iqr=q3-q1

# ll (lower limit) ul (upper limit)
ll=q1-1.5*(iqr)
ul=q3+1.5*(iqr)

ul

# balance more than upper layer (ul) are outliers

df=default[default['balance']>ul]

df

df['default'].count()

df['default'].value_counts(normalize=True).round(2)

df['default'].value_counts()

# set the data of outliers as same upperlimit

default['balance']=np.where(default['balance']>ul,ul,default['balance'])

sns.boxplot(y=default['balance'])
plt.show()

# changing the categorical value to numerical value for eg- yes, no = 1, 0

default=pd.get_dummies(default,drop_first=True)

default.head()

default.columns=['balance','income','default','student']

default.head()

from sklearn.model_selection import train_test_split

# in x taking independent variable and dropping default column in one axis
# in y taking dependent variable default only

x=default.drop('default',axis=1)
y=default['default']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=21,stratify=y)

print(x_train.shape)
print(x_test.shape)

print(y_train.value_counts(normalize=True).round(2))
print(y_test.value_counts(normalize=True).round(2))

# when we have low value of data we use SMOTE (Synthetic minority over sampling technique) method to oversample it
# it makes imbalance data equal

from imblearn.over_sampling import SMOTE
sm=SMOTE(random_state=33,sampling_strategy=0.75)
x_res,y_res=sm.fit_resample(x_train,y_train)

from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()

lr.fit(x_res,y_res)

y_pred=lr.predict(x_test)

from sklearn.metrics import confusion_matrix, classification_report

confusion_matrix(y_test,y_pred)

# Hence the accuracy is 88%

(2589+75)/(2589+75+311+25)