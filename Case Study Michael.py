# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from math import sqrt   
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor,plot_importance, plot_tree
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from prophet import Prophet


import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# # Data Preprocesing

# %% [markdown]
# **Penjelasan Varibel dataframe** \
# dfp= Dataframe Product \
# dfc= Dataframe Customer \
# dfs= Dataframe Store \
# dft= Dataframe Transaction 

# %%
dfc=pd.read_csv("D:\Pelatihan\PBA\Portofolio\Case Study Data Scientist\Case Study - Customer.csv", sep=';')
dfp=pd.read_csv("D:\Pelatihan\PBA\Portofolio\Case Study Data Scientist\Case Study - Product.csv", sep=';' )
dft=pd.read_csv("D:\Pelatihan\PBA\Portofolio\Case Study Data Scientist\Case Study - Transaction.csv", sep=';')
dfs=pd.read_csv("D:\Pelatihan\PBA\Portofolio\Case Study Data Scientist\Case Study - Store.csv", sep=';')

# %%
print(dfc.head())
print(dfc.info())
print(dfc.describe())

# %% [markdown]
# **Fitur yang akan digunakan pada table customer**
# - Fitur customerID sebagai foreign key untuk table transaksi untuk mengetahui siapa customer yang membeli
# - Fitur Income Akan dilakukan aggregration rata rata agar mengetahui berapa rata rata pendapatan pembeli untuk jumlah barang yang terjual pada hari tersebut kemungkinan nilainya akan berkorelasi
# - usia akan saya coba masukan dan nanti di akhir akan saya check korelasinya secara keseluruhan
# - Fitur berupa kategorikal akan sulit berkorelasi apabila melakukan grouping jadi tidak saya gunakan
# 

# %%
dfc.drop(['Gender','Marital Status'] , axis=1, inplace=True)

# %%
dfc.info()

# %%
#ubah income menjadi integer
dfc['Income']=dfc['Income'].str.replace(',', '')
dfc['Income']=dfc['Income'].astype('int64')

#menambah jumlah 0 sebanyak 4 digit di belakang angka
dfc['Income']=dfc['Income']*10000

# %%
dfc.info()

# %%
dfc.head()

# %%
print(dfp.head())
print(dfp.info())
print(dfp.describe())

# %% [markdown]
# **Fitur yang akan digunakan pada table product**
# - Fitur ProductID sebagai foreign key untuk table transaksi untuk mengetahui barang apa yang dibeli
# - Fitur Price Akan dilakukan aggregration rata rata atau penjumlahan agar mengetahui berapa rata rata atau penjumlahan harga barang untuk jumlah barang yang terjual pada hari tersebut kemungkinan nilainya akan berkorelasi
# - Fitur Product Name tidak akan digunakan karna berupa categorical yang sulit berkorelasi apabila dilakukan grouping

# %%
dfp.drop(['Product Name'] , axis=1, inplace=True)

# %%
dfp.info()

# %%
print(dfs.head())
print(dfs.info())
print(dfs.describe())

# %% [markdown]
# **Fitur yang akan digunakan pada table store**
# - Tidak ada fitur yang akan digunakan pada table store karna hanya mengandung nama tempat, grup store, tipe, dan lokasi secara latitude,longitude yang membuat \
# tidak ada fitur yang berkorelasi dengan jumlah barang yang terjual per hari

# %%
print(dft.head())
print(dft.info())
print(dft.describe())

# %%
dft.drop(['StoreID'], axis=1, inplace=True)

# %%
dft.head()

# %%
#rubah tipe data tanggal menjadi datetime
dft['Date']=pd.to_datetime(dft['Date'], format='%d/%m/%Y')

#menggabungkan data
df=pd.merge(dft,dfc, on='CustomerID', how='left')
df=pd.merge(df,dfp, on='ProductID', how='left')

# %%
df.info()

# %%
#hapus kolom yang tidak diperlukan
df.drop(['Price_y'], axis=1, inplace=True)
#ubah nama kolom Price_x menjadi Price
df.rename(columns={'Price_x':'Price'}, inplace=True)    

# %%
df.info()
df.head()

# %% [markdown]
# # Prophet Time Series

# %%
daily_qty = df.groupby('Date')['Qty'].sum()
daily_qty = daily_qty.sort_index()

# %%
train_size = int(0.8 * len(daily_qty))
train_data = daily_qty[:train_size]
test_data = daily_qty[train_size:]

# %%
train_data.reset_index() \
    .rename(columns={'Date':'ds',
                     'Qty':'y'}).head()


# %%
model = Prophet()
model.fit(train_data.reset_index() \
              .rename(columns={'Date':'ds',
                               'Qty':'y'}))

# %%
test_results=model.predict(df=test_data.reset_index() \
                                   .rename(columns={'Date':'ds'}))

# %%
test_results.head()

# %%
fig = model.plot_components(test_results)

# %%
# Menampilkan hasil prediksi
plt.figure(figsize=(20,8))
plt.plot(train_data.index, train_data.values, label='Data Train')
plt.plot(test_data.index, test_data.values, label='Data Test')
plt.plot(test_data.index, test_results['yhat'], label='Predicted',linestyle='dashed', color='black')
plt.xlabel('Date')
plt.ylabel('Total Quantity')
plt.title('Prophet Time Series Regression')
plt.legend()
plt.show()

# %%
print('MAE:', mean_absolute_error(test_data.values, test_results['yhat']))
print('RMSE:', sqrt(mean_squared_error(test_data.values, test_results['yhat'])))
print('MAPE:', mean_absolute_percentage_error(test_data.values, test_results['yhat']))

# %% [markdown]
# # Xgboost Time Series

# %%
dff=pd.DataFrame()
dff['date']=daily_qty.index
dff['dayofweek'] = daily_qty.index.dayofweek
dff['month'] = daily_qty.index.month
dff['dayofyear'] = daily_qty.index.dayofyear
dff['dayofmonth'] = daily_qty.index.day

dff['qty']=daily_qty.values

# %%
dff.head()

# %%
dff.info()

# %%
train_size = int(0.8 * len(daily_qty))
train_data = dff[:train_size]
test_data = dff[train_size:]

# %%
X_train = train_data.drop(['qty','date'], axis=1)
y_train = train_data['qty']
X_test = test_data.drop(['qty','date'], axis=1)
y_test = test_data['qty']

# %%
reg = XGBRegressor(n_estimators=1000)
reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        early_stopping_rounds=50,
       verbose=False)

# %%
_ = plot_importance(reg, height=0.9)

# %%
y_pred = reg.predict(X_test)

# %%
# Menampilkan hasil prediksi
plt.figure(figsize=(20,8))
plt.plot(train_data.index, train_data.values, label='Data Train')
plt.plot(test_data.index, test_data.values, label='Data Test')
plt.plot(test_data.index, y_pred, label='Predicted',linestyle='dashed', color='black')
plt.xlabel('Date')
plt.ylabel('Total Quantity')
plt.title('Xgboost Time Series Regression')
plt.legend()
plt.show()

# %%
print('MAE:', mean_absolute_error(y_test, y_pred))
print('RMSE:', sqrt(mean_squared_error(y_test, y_pred)))
print('MAPE:', mean_absolute_percentage_error(y_test, y_pred))

# %% [markdown]
# # Time Series forcasting

# %%
daily_qty = df.groupby('Date')['Qty'].sum()
daily_qty = daily_qty.sort_index()
daily_qty.index.freq = 'D'
train_size = int(0.8 * len(daily_qty))
train_data = daily_qty[:train_size]
test_data = daily_qty[train_size:]


# %%
def plot_decompose(decompose_result):
    fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1,figsize=(12,20))
    decompose_result.observed.plot(legend=False,ax=ax1,fontsize = 20,grid=True,linewidth = 3)
    ax1.set_ylabel("Observed",fontsize = 20)
    decompose_result.trend.plot(legend=False,ax=ax2,fontsize = 20,grid=True,linewidth = 3)
    ax2.set_ylabel("Trend",fontsize = 20)
    decompose_result.seasonal.plot(legend=False,ax=ax3,fontsize = 20,grid=True,linewidth = 3)
    ax3.set_ylabel("Seasonal",fontsize = 20)
    decompose_result.resid.plot(legend=False,ax=ax4,fontsize = 20,grid=True,linewidth = 3)
    ax4.set_ylabel("Residual",fontsize = 20)

# %%
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(daily_qty, period=12) 
plot_decompose(decomposition)

# %%
from pmdarima import auto_arima
# Membuat model ARIMA
stepwise_fit=auto_arima(daily_qty,trace=True,suppress_warnings=True)
stepwise_fit.summary()

# %%
from statsmodels.tsa.arima.model import ARIMA

# Membuat model ARIMA
order = (15,2,1)  # Nilai ini didapat dari pengujian autoarima terbaik
model = ARIMA(train_data, order=order)
model_fit = model.fit()

# Melakukan prediksi pada data uji
predictions = model_fit.forecast(steps=len(test_data))

# %%
# Menampilkan hasil prediksi
plt.figure(figsize=(20,8))
plt.plot(train_data.index, train_data.values, label='Data Train')
plt.plot(test_data.index, test_data.values, label='Data Test')
plt.plot(test_data.index, predictions, label='Predicted',linestyle='dashed', color='black')
plt.xlabel('Date')
plt.ylabel('Total Quantity')
plt.title('ARIMA Time Series Regression')
plt.legend()
plt.show()

# %%
print('MAE:', mean_absolute_error(test_data.values, predictions))
print('RMSE:', sqrt(mean_squared_error(test_data.values, predictions)))
print('MAPE:', mean_absolute_percentage_error(test_data.values, predictions))

# %% [markdown]
# # Random Forest Regression

# %%
aggregation = {
    'Qty': 'sum',
    'Price': 'sum',
    'TotalAmount': 'sum',
    'Income': 'mean',
    'Age': 'mean'  
}
time_series_data=df.groupby('Date').agg(aggregation).reset_index()
time_series_data.index.freq = 'D'
time_series_data.head()

# %%
#membulatkan umur dan icome menjadi integer
time_series_data['Age']=time_series_data['Age'].astype('int64')
time_series_data['Income']=time_series_data['Income'].astype('int64')

# %%
time_series_data.info()

# %%
time_series_data.head()

# %%
#melihat korelasi antar variabel
plt.figure(figsize=(10,8))
sns.heatmap(time_series_data.corr(), annot=True)
plt.show()

# %% [markdown]
# **Hasil Korelasi** \
# Berdasarkan Heatmap korelasi diatas variabel Quantity berkorelasi kuat positif terhadap harga dan jumlah amount \
# Serta memiliki korelasi lemah terhadap umur dan hampir tidak memiliki korelasi terhadap pendapatan \
# jadi saya akan mencoba 2x pengetesan algoritma arima mengikutkan fitur umur dan tidak mengikutkan fitur umur

# %%
time_series_data.drop(['Income'], axis=1, inplace=True)

# %%
time_series_data['Day']=time_series_data.index

# %%
time_series_data.info()

# %%
train_size = int(0.8 * len(time_series_data))
train = time_series_data[:train_size]
test = time_series_data[train_size:]

# %%
X_train = train.drop(['Qty','Date'], axis=1)
y_train = train['Qty']
X_test = test.drop(['Qty','Date'], axis=1)
y_test = test['Qty']

rf = RandomForestRegressor(n_estimators=1000, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print('MAE:', mean_absolute_error(y_test, y_pred))
print('RMSE:', sqrt(mean_squared_error(y_test, y_pred)))

# %%
# Menampilkan hasil prediksi
plt.figure(figsize=(10,8))
plt.plot(test_data.index, y_test, label='Actual')
plt.plot(test_data.index, y_pred, label='Predicted',linestyle='dashed', color='red')
plt.xlabel('Date')
plt.ylabel('Total Quantity')
plt.title('Random Forest Time Series Regression')
plt.legend()
plt.show()

# %% [markdown]
# # Clustering with KNN Clustering

# %%
aggregation = {
    'TransactionID': 'count',
    'Qty': 'sum',
    'TotalAmount': 'sum'
}
cluster_data = df.groupby('CustomerID').agg(aggregation).reset_index()
cluster_data.head()

# %%
cluster_data.drop(['CustomerID'], axis=1, inplace=True)

# %%
scaler=StandardScaler()
X=scaler.fit_transform(cluster_data)

# %%
wcss=[]
for n in range(1,11):
    kmeans=KMeans(n_clusters=n, init='k-means++', n_init=10, max_iter=300, random_state=100)
    kmeans.fit(cluster_data)
    wcss.append(kmeans.inertia_)

# %%
plt.figure(figsize=(10,8))
plt.plot(range(1,11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# %%
num_clusters = 4
kmeans = KMeans(n_clusters=num_clusters, init='k-means++', n_init=10, max_iter=300, random_state=100)
kmeans.fit(cluster_data)
cluster_labels=kmeans.fit_predict(X)
cluster_data['Cluster'] = cluster_labels


# %%
cluster_data.head()

# %%
cluster_data.value_counts('Cluster')

# %%
plt.figure(figsize=(10,8))
sns.scatterplot(data=cluster_data, x='TransactionID', y='Qty', hue='Cluster', palette='Set1')
plt.show()

# %%
plt.figure(figsize=(10,8))
sns.scatterplot(data=cluster_data, x='TransactionID', y='TotalAmount', hue='Cluster', palette='Set1')
plt.show()

# %%
plt.figure(figsize=(10,8))
sns.scatterplot(data=cluster_data, x='Qty', y='TotalAmount', hue='Cluster', palette='Set1')
plt.show()


