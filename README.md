# 基於深度學習及機器學習的人類活動識別預測
## Overview
Human Activity Recognition(HAV) 已經成為過去幾年的一個重要研究領域，並且越來越受到普及計算研究界的關注，那我們可以觀測的不只是一些平常的運動，例如坐下、慢跑、站立，當然也可能是一些工作領域，例如搬運貨物，廚房炒菜，醫生開刀等等，且HAR也能顧及未來許多重要的發展，例如：老人長照、健身姿勢是否不良等。目前市面上的智慧型手機都有一個三軸的加速度計，可以測量三個空間維度的加速度，這使得感測數據更容易地去取得，此外感測數據本身是需要遠端來記錄的，這涉及到了物聯網的感知層以及網路層，而現在處理的是物聯網的應用層面，藉由此應用層面我們來發揮其最大效益。
## Background
HAR 在傳統上多用於區分不同的活動，亦即在同一個時間點執行什麼活動，而此數據集為舉重練習數據集，是用來調查、檢視受測者運動姿勢是否正確，使得運動者能達到最佳的訓練效果。數據集取自六名參與者的腰帶、手腕、手臂、啞鈴，並以五種不同的方式進行一組 10 次的單側啞鈴舉。
## Data Exploration
本次使用的程式碼為 python，首先我們先將 dataset 利用 pandas 套件讀取，接著因為我們沒有要使用 RNN 或是 LSTM(GRU)，所以把有關時間序列的數據都刪除
```
train2 = train.drop(['Unnamed: 0', 'user_name','raw_timestamp_part_1','raw_timestamp_part_2','cvtd_timestamp','new_window','num_window'], axis=1)

``` 
刪除任何有 NaN 的行
```
train2 = train2.dropna(axis=1, how='any')
``` 
畫出五種活動數據的直方圖，看是否有類別數據過多或是過少
```
train2['classe'].value_counts().plot(kind='bar',
                                   title='Training Examples by Activity Type')
plt.show()
``` 
![image](https://github.com/03053020ITE/ship-detection/blob/master/show/7.PNG)

因為要放入深度學習進去訓練，因此把 label 轉換為 1-hot-encoding
A 轉換為 10000 ； B 轉換為 01000；C 轉換為 00100；D 轉換為 00010；E 轉換為 00001
```
train3 = pd.get_dummies(data=train2, columns=["classe"])
train3[:2]
``` 
將訓練數據拆分為訓練集及驗證集，方便訓練時查看不要 overfitting
```
msk = np.random.rand(len(train2)) < 0.8
train_df = train3[msk]
test_df = train3[~msk]
``` 
使用 values 將原本的 dataframe 數據強制轉換成 numpy 格式的數據
```
nparray_train = train_df.values
nparray_test = test_df.values
``` 
把前 52 行數據當成 feature ，最後 5 行數據當成 label
```
train_label = nparray_train[:, 52:57]
test_label = nparray_test[:, 52:57]
train_feature = nparray_train[:, 0:52]
test_feature = nparray_test[:, 0:52]
```
使用正規化及標準化測試準確率
```
from sklearn import preprocessing
# 正規化
minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
train_feature=minmax_scale.fit_transform(train_feature)
test_feature=minmax_scale.fit_transform(test_feature)
# 標準化
scaler = preprocessing.StandardScaler()
train_feature = scaler.fit_transform(train_feature)
test_feature = scaler.fit_transform(test_feature)
```
在神經網路架設部分，使用了四層 Dense 層，且加入 Dropout
![image](https://github.com/03053020ITE/ship-detection/blob/master/show/7.PNG)