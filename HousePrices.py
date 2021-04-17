# Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing



df_train = pd.read_csv("train.csv")
#df = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")
  
#removing null columns with more than 50% missing values
#visualizing the missing values
def missingdata(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    ms=pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    ms= ms[ms["Percent"] > 0]
    f,ax =plt.subplots(figsize=(8,6))
    plt.xticks(rotation='90')
    sns.barplot(ms.index, ms["Percent"],color="green",alpha=0.8)
    plt.xlabel('Features', fontsize=15)
    plt.ylabel('Percent of missing values', fontsize=15)
    plt.title('Percent missing data by feature', fontsize=15)
    return ms

missingdata(df_train)
missingdata(df_test)
percent_train = (df_train.isnull().sum()/df_train.isnull().count()*100).sort_values(ascending = False)
percent_test = (df_test.isnull().sum()/df_test.isnull().count()*100).sort_values(ascending = False)
drop_col=["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu"]
df_train = df_train.drop(drop_col, axis = 1)
df_test = df_test.drop(drop_col, axis = 1)

final = [df_train,df_test]

percent_train = (df_train.isnull().sum()/df_train.isnull().count()*100).sort_values(ascending = False)
percent_test = (df_test.isnull().sum()/df_test.isnull().count()*100).sort_values(ascending = False)
miss_train = percent_train.index[percent_train>0]
miss_test = percent_test.index[percent_test>0] 

for i in miss_train:
  df_train[i].fillna(df_train[i].mode()[0], inplace = True)

for j in miss_test:
   df_test[j].fillna(df_test[j].mode()[0], inplace = True)

label_encoder = preprocessing.LabelEncoder() 

cat = []
for l in df_train:
  if (isinstance(df_train[l][0],str)):
    cat.append(l)
    df_train[l]= label_encoder.fit_transform(df_train[l]) 
    df_train[l].unique()

cor = df_train.corr()
cor_target = abs(cor["SalePrice"]) 
rel_features = cor_target[cor_target > 0.30]

rel_col = [rel_features.index]

cat_corr = []

for k in rel_features.index:
    for m in cat:
      if k==m:
        cat_corr.append(k)
    

for i in rel_col:
   df_train_fin = df_train[i]

cat = []
for l in df_train_fin:
  if (isinstance(df_train_fin[l][0],str)):
    cat.append(l)


y = df_train['SalePrice'].values
df_train_fin = df_train_fin.drop('SalePrice', axis = 1)

rel_col_test = [df_train_fin.columns]

for i in rel_col_test:
   df_test_fin = df_test[i]


df_train_fin = pd.get_dummies(df_train_fin, columns = cat_corr,prefix = cat_corr, drop_first = True)  
df_test_fin =  pd.get_dummies(df_test_fin, columns = cat_corr,prefix = cat_corr, drop_first = True)


X = df_train_fin
X_test = df_test_fin

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
X_test = sc_X.fit_transform(X_test)
y = sc_y.fit_transform(y.reshape(-1,1))


#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)



# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)


y_pred = regressor.predict(X_test)
y_pred = sc_y.inverse_transform(y_pred)
#y_test = sc_y.inverse_transform(y_test)


#XGBoost
from xgboost import XGBRegressor
XG_reg = XGBRegressor()
XG_reg.fit(X, y)

y_pred_XG = XG_reg.predict(X_test)
y_pred_XG = sc_y.inverse_transform(y_pred_XG)
