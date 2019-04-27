# 基於深度學習及機器學習的人類活動識別預測
## Overview
Human Activity Recognition(HAV) 已經成為過去幾年的一個重要研究領域，並且越來越受到研究界的關注，那我們可以觀測的不只是一些平常的運動，例如坐下、慢跑、站立，當然也可能是一些工作領域，例如搬運貨物，廚房炒菜，醫生開刀等等，且HAR也能顧及未來許多重要的發展，例如：老人長照、健身姿勢是否不良等。目前市面上的智慧型手機都有一個三軸的加速度計，可以測量三個空間維度的加速度，這使得感測數據更容易地去取得，此外感測數據本身是需要遠端來記錄的，這涉及到了物聯網的感知層以及網路層，而現在處理的是物聯網的應用層面，藉由此應用層面我們來發揮其最大效益。
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
train2['classe'].value_counts().plot(kind='bar',title='Training Examples by Activity Type')
plt.show()
``` 
![image](https://github.com/03053020ITE/human-activity-recognition/blob/master/image/act_type.PNG)

因為要放入深度學習進去訓練，因此把 label 轉換為 1-hot-encoding，轉換完成後，A 會轉換為 10000 ； B 會轉換為 01000；C 會轉換為 00100；D 會轉換為 00010；E 會轉換為 00001
```
train3 = pd.get_dummies(data=train2, columns=["classe"])
train3[:2]
``` 
![image](https://github.com/03053020ITE/human-activity-recognition/blob/master/image/one_hot.PNG)

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
![image](https://github.com/03053020ITE/human-activity-recognition/blob/master/image/numpy_values.PNG)

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
## Prediction Modeling
#### 使用深度學習 (Dense)
神經網路架設了六層 Dense 層，且加入 Dropout 防止過度擬合，更利用 BatchNormalization來做正規化 那因為模型中的參數愈小代表模型愈簡單，愈不容易產生過擬合現象，那神經元參數配置上，我第一層的神經元為 256，第二層的神經元為 256，第三層的神經元為 128，第四層的神經元為 64，第五層的神經元為 32，第六層的神經元為 16，第七層的神經元為 5，中間的Dorpout 設置為 0.2，也就是經過一層後自動拋棄 20% 神經元，保留 80% 神經元

```
model = Sequential()
model.add(Dense(256, kernel_initializer='normal',activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(256, kernel_initializer='normal',activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(128, kernel_initializer='normal',activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(64, kernel_initializer='normal',activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(32, kernel_initializer='normal',activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(16, kernel_initializer='normal',activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(5, activation = 'softmax'))
```
![image](https://github.com/03053020ITE/human-activity-recognition/blob/master/image/nn1.PNG)

![image](https://github.com/03053020ITE/human-activity-recognition/blob/master/image/nn2.PNG)
#### 使用深度學習 (CNN)
神經網路架設了 convolution1D ，在數據處理部分，將 52 行的數據切割成 4 份，並放入分別放入 4 個維度，最後把這四個維度當作一張照片放入 CNN 裡頭

![image](https://github.com/03053020ITE/human-activity-recognition/blob/master/image/CNN.PNG)

#### 使用機器學習
隨機深林分類器(RandomForestClassifier)進行模型訓練以及預測分析
```
from sklearn.ensemble import RandomForestClassifier, 
RFC = RandomForestClassifier()
RFC.fit(train_feature,train_label)
```
梯度提升決策樹(GradientBoostingClassifie)進行整合模型的訓練以及預測分析
```
from sklearn.ensemble import GradientBoostingClassifier
GBC = GradientBoostingClassifier()
GBC.fit(train_feature, train_label)
```
使用單一決策樹(DecisionTreeClassifier)進行模型訓練以及預測分析
```
from sklearn.tree import DecisionTreeClassifier
DTC = DecisionTreeClassifier()
DTC.fit(train_feature, train_label)
```
使用基於線性假設的支援向量機(SVM)進行模型訓練以及預測分析
```
from sklearn.svm import SVC
SVCL = SVC(kernel='linear')
SVCL.fit(train_feature, train_label)
```
使用基於多項式 kernel 的支援向量機(SVM)分類器進行模型訓練以及預測分析
```
from sklearn.svm import SVC
SVCP = SVC(kernel='poly')
SVCP.fit(train_feature, train_label)
```
使用基於高斯 kernel 的支援向量機(SVM)分類器進行模型訓練以及預測分析
```
from sklearn.svm import SVC
SVCR = SVC(kernel='rbf')
SVCR.fit(train_feature, train_label)
```
使用基於 sigmoid kernel 的支援向量機(SVM)分類器進行模型訓練以及預測分析
```
from sklearn.svm import SVC
SVCS = SVC(kernel='sigmoid')
SVCS.fit(train_feature, train_label)
```
使用高斯樸素貝葉斯(Gaussian Naive Bayes)分類器進行模型訓練以及預測分析
```
from sklearn.naive_bayes import GaussianNB
GNB = GaussianNB()
GNB.fit(train_feature,train_label)
```
使用多項式樸素貝葉斯(Multinomial Naive Bayes)分類器進行模型訓練以及預測分析
```
from sklearn.naive_bayes import MultinomialNB
MNB = MultinomialNB()
MNB.fit(train_feature,train_label)
```
使用伯努力樸素貝葉斯(Bernoulli Naive Bayes)分類器進行模型訓練以及預測分析
```
from sklearn.naive_bayes import BernoulliNB
BNB = BernoulliNB()
BNB.fit(train_feature,train_label)
```
使用邏輯迴歸(LogisticRegression)進行模型訓練以及預測分析
```
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(train_feature, train_label)
```
使用 KNN 進行模型訓練以及預測分析
```
from sklearn.neighbors import KNeighborsClassifier
KNC = KNeighborsClassifier()
KNC.fit(train_feature,train_label)
```

## Model Application
### 深度學習(Dense)
訓練集準確率、驗證集準確率

![image](https://github.com/03053020ITE/human-activity-recognition/blob/master/image/dense_acc.PNG)

將切分的驗證集放入驗證準確率、預測test題目的 20 人答案

![image](https://github.com/03053020ITE/human-activity-recognition/blob/master/image/dense_test.PNG)
### 深度學習(CNN)
訓練集準確率、驗證集準確率

![image](https://github.com/03053020ITE/human-activity-recognition/blob/master/image/RFC_A.PNG)

將切分驗證集放入驗證準確率、預測結果為題目的 20 人答案

![image](https://github.com/03053020ITE/human-activity-recognition/blob/master/image/RFC_A.PNG)

以下我會將數據進行三種規則，第一種為沒有正規化也沒有標準化；第二種為有正規化沒有標準化；第三種為有標準化沒有正規化。最後都是顯示最好的準確率之混淆矩陣，此混淆矩陣、準確率為隨機切割的驗證集所算出，預測結果為題目的 20 人答案
### 隨機森林分類器
隨機森林分類器方面，如果將沒有正規化、標準化的數據放入得到的準確率為 0.9900，只進行正規化轉換後得到的準確率為 0.7323，只進行標準化轉換後得到的準確率為 0.9353

![image](https://github.com/03053020ITE/human-activity-recognition/blob/master/image/RFC_A.PNG)
### 輸出隨機森林分類器在測試集上的分類準確性，以及更加詳細的精確率、召回率、F1指標
![image](https://github.com/03053020ITE/human-activity-recognition/blob/master/image/RFC_B.PNG)

### 梯度提升決策樹
梯度提升決策樹方面，如果將沒有正規化、標準化的數據放入得到的準確率為 0.9697，只進行正規化轉換後得到的準確率為 0.7321，只進行標準化轉換後得到的準確率為 0.8586

![image](https://github.com/03053020ITE/human-activity-recognition/blob/master/image/GBC_A.PNG)
### 輸出梯度提升決策樹在測試集上的分類準確性，以及更加詳細的精確率、召回率、F1指標
![image](https://github.com/03053020ITE/human-activity-recognition/blob/master/image/GBC_B.PNG)

### 單一決策樹
單一決策樹方面，如果將沒有正規化、標準化的數據放入得到的準確率為 0.9659，只進行正規化轉換後得到的準確率為 0.4489，只進行標準化轉換後得到的準確率為 0.8175

![image](https://github.com/03053020ITE/human-activity-recognition/blob/master/image/DTC_A.PNG)
### 輸出單一決策樹在測試集上的分類準確性，以及更加詳細的精確率、召回率、F1指標
![image](https://github.com/03053020ITE/human-activity-recognition/blob/master/image/DTC_B.PNG)

### 線性假設的支援向量機
線性假設的支援向量機方面，如果將沒有正規化、標準化的數據放入得到的準確率為 0.，只進行正規化轉換後得到的準確率為 0.4810，只進行標準化轉換後得到的準確率為 0.7578

![image](https://github.com/03053020ITE/human-activity-recognition/blob/master/image/SVCL_A.PNG)
### 輸出線性假設的支援向量機在測試集上的分類準確性，以及更加詳細的精確率、召回率、F1指標
![image](https://github.com/03053020ITE/human-activity-recognition/blob/master/image/SVCL_B.PNG)

### 多項式 kernel 的支援向量機
多項式 kernel的支援向量機方面，如果將沒有正規化、標準化的數據放入得到的準確率為 0.9889，只進行正規化轉換後得到的準確率為 0.4285，只進行標準化轉換後得到的準確率為 0.9485

![image](https://github.com/03053020ITE/human-activity-recognition/blob/master/image/SVCP_A.PNG)
### 輸出多項式 kernel 的支援向量機在測試集上的分類準確性，以及更加詳細的精確率、召回率、F1指標
![image](https://github.com/03053020ITE/human-activity-recognition/blob/master/image/SVCP_B.PNG)

### 高斯 kernel 的支援向量機
高斯 kernel 的支援向量機方面，如果將沒有正規化、標準化的數據放入得到的準確率為 0.2826，只進行正規化轉換後得到的準確率為 0.6075，只進行標準化轉換後得到的準確率為 0.9513

![image](https://github.com/03053020ITE/human-activity-recognition/blob/master/image/SVCR_A.PNG)
### 輸出高斯 kernel 的支援向量機在測試集上的分類準確性，以及更加詳細的精確率、召回率、F1指標
![image](https://github.com/03053020ITE/human-activity-recognition/blob/master/image/SVCR_B.PNG)

### sogmoid kernel 的支援向量機
sogmoid kernel 的支援向量機方面，如果將沒有正規化、標準化的數據放入得到的準確率為 0.1692，只進行正規化轉換後得到的準確率為 0.5826，只進行標準化轉換後得到的準確率為 0.3491

![image](https://github.com/03053020ITE/human-activity-recognition/blob/master/image/SVCS_A.PNG)
### 輸出 sogmoid kernel 的支援向量機在測試集上的分類準確性，以及更加詳細的精確率、召回率、F1指標
![image](https://github.com/03053020ITE/human-activity-recognition/blob/master/image/SVCS_B.PNG)

### 高斯樸素貝葉斯分類器
高斯樸素貝葉斯分類器方面，如果將沒有正規化、標準化的數據放入得到的準確率為 0.4960，只進行正規化轉換後得到的準確率為 0.2847，只進行標準化轉換後得到的準確率為 0.3538

![image](https://github.com/03053020ITE/human-activity-recognition/blob/master/image/GNB_A.PNG)
### 輸出高斯樸素貝葉斯分類器測試集上的分類準確性，以及更加詳細的精確率、召回率、F1指標
![image](https://github.com/03053020ITE/human-activity-recognition/blob/master/image/GNB_B.PNG)

### 多項式樸素貝葉斯分類器
多項式樸素貝葉斯分類器方面，如果將沒有正規化、標準化的數據是無法放入得到的準確率的，因為輸入不能為負數，只進行正規化轉換後得到的準確率為 0.3499，只進行標準化轉換依舊是負數，因此也無法求得準確率

![image](https://github.com/03053020ITE/human-activity-recognition/blob/master/image/MNB_A.PNG)
### 輸出多項式樸素貝葉斯分類器測試集上的分類準確性，以及更加詳細的精確率、召回率、F1指標
![image](https://github.com/03053020ITE/human-activity-recognition/blob/master/image/MNB_B.PNG)

### 伯努力樸素貝葉斯分類器
伯努力樸素貝葉斯分類器方面，如果將沒有正規化、標準化的數據放入得到的準確率為 0.4391，只進行正規化轉換後得到的準確率為 0.2972，只進行標準化轉換後得到的準確率為 0.4256

![image](https://github.com/03053020ITE/human-activity-recognition/blob/master/image/BNB_A.PNG)
### 輸出伯努力樸素貝葉斯分類器測試集上的分類準確性，以及更加詳細的精確率、召回率、F1指標
![image](https://github.com/03053020ITE/human-activity-recognition/blob/master/image/BNN_B.PNG)

### 邏輯迴歸
邏輯迴歸方面，如果將沒有正規化、標準化的數據放入得到的準確率為 0.7389，只進行正規化轉換後得到的準確率為 0.5621，只進行標準化轉換後得到的準確率為 0.7133

![image](https://github.com/03053020ITE/human-activity-recognition/blob/master/image/LR_A.PNG)
### 輸出邏輯迴歸測試集上的分類準確性，以及更加詳細的精確率、召回率、F1指標
![image](https://github.com/03053020ITE/human-activity-recognition/blob/master/image/LR_B.PNG)

### 最近鄰居法
最近鄰居法方面，如果將沒有正規化、標準化的數據放入得到的準確率為 0.9189，只進行正規化轉換後得到的準確率為 0.9359，只進行標準化轉換後得到的準確率為 0.9761

![image](https://github.com/03053020ITE/human-activity-recognition/blob/master/image/KNC_A.PNG)
### 輸出最近鄰居法測試集上的分類準確性，以及更加詳細的精確率、召回率、F1指標
![image](https://github.com/03053020ITE/human-activity-recognition/blob/master/image/KNC_B.PNG)
