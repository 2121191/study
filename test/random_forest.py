import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# 读取训练和测试数据集
train_data = pd.read_csv('../data/train.csv')
test_data = pd.read_csv('../data/test.csv')

# 处理缺失值（训练数据）
train_data['Age'] = train_data.groupby(['Sex', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.mean()))  # 填充Age
train_data['Fare'] = train_data['Fare'].fillna(train_data['Fare'].mean())  # 填充Fare
train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])  # 填充Embarked

# 处理缺失值（测试数据）
test_data['Age'] = test_data.groupby(['Sex', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.mean()))  # 填充Age
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].mean())  # 填充Fare
test_data['Embarked'] = test_data['Embarked'].fillna(test_data['Embarked'].mode()[0])  # 填充Embarked

# 特征工程（训练数据）
train_data['Title'] = train_data['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())  # 提取称谓
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1  # 计算家庭大小
train_data['AgeGroup'] = pd.cut(train_data['Age'], bins=[0, 12, 18, 30, 50, 80], labels=['Child', 'Teen', 'YoungAdult', 'Adult', 'Senior'])

# 特征工程（测试数据）
test_data['Title'] = test_data['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())  # 提取称谓
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1  # 计算家庭大小
test_data['AgeGroup'] = pd.cut(test_data['Age'], bins=[0, 12, 18, 30, 50, 80], labels=['Child', 'Teen', 'YoungAdult', 'Adult', 'Senior'])

# 对类别特征进行独热编码（训练数据）
train_data_encoded = pd.get_dummies(train_data[['Sex', 'Embarked', 'Pclass', 'AgeGroup', 'Title', 'FamilySize']], drop_first=True)

# 目标变量
y_train = train_data['Survived']

# 特征变量
X_train = train_data_encoded

# 对测试数据进行相同的独热编码，但确保列名与训练数据一致
test_data_encoded = pd.get_dummies(test_data[['Sex', 'Embarked', 'Pclass', 'AgeGroup', 'Title', 'FamilySize']], drop_first=True)

# 找出训练集和测试集中的列差异
train_columns = X_train.columns
test_columns = test_data_encoded.columns

# 找出训练集中有，但测试集中没有的列
missing_cols = set(train_columns) - set(test_columns)
# 找出测试集中有，但训练集中没有的列
extra_cols = set(test_columns) - set(train_columns)

# 对测试集补充缺失的列，并设置为0
for col in missing_cols:
    test_data_encoded[col] = 0

# 删除测试集中额外的列
test_data_encoded = test_data_encoded.drop(columns=extra_cols)

# 确保列的顺序一致
test_data_encoded = test_data_encoded[train_columns]

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(test_data_encoded)

# 创建随机森林模型
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 训练模型
model.fit(X_train_scaled, y_train)

# 使用训练好的模型进行预测
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

# 输出训练集准确率
print("Training Accuracy:", accuracy_score(y_train, y_pred_train))

# 输出测试集预测结果到CSV文件
output = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': y_pred_test})
output.to_csv('titanic_predictions_rf.csv', index=False)
print("Predictions saved to 'titanic_predictions_rf.csv'")

# 读取实际的生存结果（gender_submission.csv）
gender_submission_path = '../data/gender_submission.csv'
gender_submission = pd.read_csv(gender_submission_path)

# 读取模型的预测结果（titanic_predictions_rf.csv）
predictions_path = 'titanic_predictions_rf.csv'
predictions = pd.read_csv(predictions_path)

# 合并两个数据集，按PassengerId对齐
merged_data = pd.merge(predictions, gender_submission, on='PassengerId')

# 计算准确率
accuracy = accuracy_score(merged_data['Survived_x'], merged_data['Survived_y'])
print(f"Final Accuracy on Test Set: {accuracy:.4f}")

# 用classification_report进行更详细的评估
print(classification_report(merged_data['Survived_x'], merged_data['Survived_y']))
