import pandas as pd
import numpy as np
from tensorflow.keras.layers import Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns

#%% 读取数据集
dataset_path = '../data/train.csv'
data = pd.read_csv(dataset_path)

# 处理缺失值
data['Age'] = data.groupby(['Sex', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.mean()))  # 根据性别和舱位填充Age
data['Fare'] = data['Fare'].fillna(data['Fare'].mean())  # 使用Fare的均值填充缺失值
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])  # 使用Embarked的众数填充缺失值

# 特征工程
data['Title'] = data['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())  # 提取称谓
data['FamilySize'] = data['SibSp'] + data['Parch'] + 1  # 计算家庭大小
data['AgeGroup'] = pd.cut(data['Age'], bins=[0, 12, 18, 30, 50, 80], labels=['Child', 'Teen', 'YoungAdult', 'Adult', 'Senior'])

# 对类别特征进行独热编码
data_encoded = pd.get_dummies(data[['Sex', 'Embarked', 'Pclass', 'AgeGroup', 'Title', 'FamilySize']], drop_first=True)

# 目标变量
y = data['Survived']

# 特征变量
X = data_encoded

#%% 特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#%% 将数据转换为适合CNN输入的格式
X_scaled_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)  # 转换为 (样本数, 特征数, 1)

#%% 划分数据集：80%用于训练，20%用于测试
X_train, X_test, y_train, y_test = train_test_split(X_scaled_reshaped, y, test_size=0.2, random_state=42)

#%% 创建CNN模型
model = Sequential()

# 输入层（显式地定义输入形状）
model.add(Input(shape=(X_train.shape[1], 1)))  # 这里我们显式地定义输入的形状

# 第一层卷积层
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))  # 池化层

# 第二层卷积层
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))

# 展平层
model.add(Flatten())

# 全连接层
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

# 输出层（sigmoid激活函数输出存活概率）
model.add(Dense(1, activation='sigmoid'))  # 使用sigmoid激活函数输出存活的概率

# 编译模型
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

#%% 训练模型
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test), verbose=2)

#%% 预测存活的概率
y_prob = model.predict(X_test)  # 获取预测的存活概率

# 显示前10个乘客的存活概率
print("Predicted survival probabilities for the first 10 passengers:", y_prob[:10])

# 根据概率值判断存活与否（例如：大于0.5预测存活，小于0.5预测未存活）
y_pred = (y_prob > 0.5).astype("int32")  # 二分类结果
print("Predicted survival (0=not survived, 1=survived) for the first 10 passengers:", y_pred[:10])

#%% 评估模型性能
print("Accuracy:", accuracy_score(y_test, y_pred))  # 输出准确率
print(classification_report(y_test, y_pred))  # 输出分类报告

# 绘制训练过程中的损失和准确率
plt.figure(figsize=(12, 5))

# 绘制损失曲线
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 绘制准确率曲线
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

#%% 保存模型
model.save('titanic_cnn_model.keras')

# 加载模型
# model = tf.keras.models.load_model('titanic_cnn_model.keras')
