#%%
import pandas as pd
dataset_path = '../data/train.csv'
data = pd.read_csv(dataset_path)
print(data.head())

#%%
# 输出数据集的行数和列数
num_rows, num_cols = data.shape
print("数据集行数:", num_rows)
print("数据集列数:", num_cols)
# 输出数据集的基本信息
print("\n数据集的基本信息：")
print(data.info())

#%%
# 检查每列的缺失值数量
missing_values = data.isnull().sum()
print("\n缺失值数量：")
print(missing_values)

#%%
#缺失信息补全
data['Age'] = data.groupby(['Sex', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.mean()))
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])  # 填充Embarked

#%%
data['date'] = 19120410
data['date'] = pd.to_datetime(data['date'])

#%%
# 提取Title
data['Title'] = data['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
# 年龄组
data['AgeGroup'] = pd.cut(data['Age'], bins=[0, 12, 18, 30, 50, 80], labels=['Child', 'Teen', 'YoungAdult', 'Adult', 'Senior'])
# 票价组
data['FareGroup'] = pd.cut(data['Fare'], bins=[0, 10, 50, 100, 500], labels=['Low', 'Medium', 'High', 'VeryHigh'])

#%%
print("处理后的数据集信息：")
print(data.info())
#%%
import seaborn as sns
import matplotlib.pyplot as plt

# 设置plot格式
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

palette = sns.color_palette("Set2", 6)


#%%
#统计不同性别的生存数量
gender_survived = data.groupby(['Sex', 'Survived']).size().unstack(fill_value=0)
gender_survived.plot(kind='bar', stacked=True, color=['red', 'green'], figsize=(8, 6))
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Survival Counts by Gender')
plt.xticks(rotation=0)
plt.legend(['Not Survived', 'Survived'], loc='upper left')
plt.tight_layout()
plt.show()


#%%
 # 使用箱线图比较不同舱位等级的乘客年龄分布
plt.boxplot([data[data['Pclass'] == 1]['Age'].dropna(),
             data[data['Pclass'] == 2]['Age'].dropna(),
             data[data['Pclass'] == 3]['Age'].dropna()])
plt.xlabel('Pclass')
plt.ylabel('Age')
plt.title('Age Distribution by Passenger Class')
plt.xticks([1, 2, 3], ['1st Class', '2nd Class', '3rd Class'])
plt.show()


#%%
#统计乘客的登船港口分布
embarked_count = data['Embarked'].value_counts()
colors = sns.color_palette("Set2", len(embarked_count))
# 绘制饼图
plt.figure(figsize=(8, 8))  # 设置图形大小
plt.pie(embarked_count, labels=embarked_count.index, autopct='%1.1f%%', startangle=90, colors=colors, wedgeprops={'edgecolor': 'black'})

plt.title('Distribution of Passengers by Embarked')

plt.axis('equal')
plt.show()

#%%
#统计不同年龄段乘客的生存情况
age_groups = pd.cut(data['Age'], bins=[0, 12, 18, 30, 50, 80])
survival_by_age = data.groupby(age_groups, observed=False)['Survived'].mean().reset_index()
plt.figure(figsize=(8, 8))
plt.pie(survival_by_age['Survived'], labels=survival_by_age['Age'].astype(str), autopct='%1.1f%%',
        startangle=90, colors=palette, wedgeprops={'edgecolor': 'black'})
plt.title('Survival Rate by Age Group')
plt.axis('equal')
plt.tight_layout()
plt.show()

#%%
#统计票价的分布
plt.figure()
sns.histplot(data['Fare'], kde=True, color=palette[3])
plt.title('Distribution of Fare')
plt.xlabel('Fare')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

#%%
#统计不同票价的生存状况
sns.boxplot(x='FareGroup', y='Fare', hue='Survived', data=data)
plt.title('Fare Distribution by Fare Group and Survival Status')
plt.show()


#%%
# 统计不同舱位等级的乘客生存率
pclass_survived = data.groupby('Pclass')['Survived'].mean()
plt.figure()
pclass_survived.plot(kind='bar', color=palette[4])
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')
plt.title('Survival Rate by Passenger Class')
plt.xticks([0, 1, 2], ['1st Class', '2nd Class', '3rd Class'], rotation=0)
plt.tight_layout()
plt.show()

#%%
#统计不同舱位等级的乘客数量
sns.countplot(x='Pclass', data=data)
plt.title("Number of passengers in different cabin classes")
plt.show()

#%%
# 统计不同船舱等级下男性和女性乘客数量
gender_counts_by_pclass = data.groupby(['Pclass', 'Sex']).size().unstack()
gender_counts_by_pclass.plot(kind='bar', stacked=True, color=[palette[0], palette[1]], figsize=(8, 6))
plt.xlabel('Passenger Class')
plt.ylabel('Count')
plt.title('Passenger Count by Pclass and Sex')
plt.tight_layout()
plt.show()

#%%
#统计存活的男性与女性的平均年龄
average_age_survived = data.groupby('Sex')['Age'].mean()

# 设置颜色：女性用粉色 男性用蓝色（绝非性别刻板印象QAQ）
colors = ['#6699FF' if sex == 'male' else 'pink' for sex in average_age_survived.index]

# 绘制柱状图显示存活的男性和女性的平均年龄
plt.bar(average_age_survived.index, average_age_survived, color=colors)

plt.xlabel('Sex')
plt.ylabel('Average Age')
plt.title('Average Age of Survived Passengers by Sex')
plt.show()

#%%
#统计家庭规模
data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
passenger_count_by_family_size = data['FamilySize'].value_counts()
plt.figure()
passenger_count_by_family_size.plot(kind='bar', color=palette[5])
plt.xlabel('Family Size')
plt.ylabel('Count')
plt.title('Passenger Count by Family Size')
plt.tight_layout()
plt.show()

#%%
#统计生存率与称谓的关系（不同称谓反映不同身份）
title_survival_rate = data.groupby('Title')['Survived'].mean()
plt.figure()
title_survival_rate.plot(kind='bar', color=palette[2])
plt.title('Survival Rate by Title')
plt.ylabel('Survival Rate')
plt.xlabel('Title')
plt.tight_layout()
plt.show()


#%%
#统计不同年龄段和船舱等级组合下乘客存活状况

# 计算不同年龄段和船舱等级组合下的乘客存活率
survival_rate_by_age_group_pclass = data.groupby(['Age', 'Pclass'])['Survived'].mean().unstack()

# 格式化y轴的标签（即年龄段）
survival_rate_by_age_group_pclass.index = survival_rate_by_age_group_pclass.index.map(lambda x: f'{x:.2f}')

# 调整图形大小
plt.figure(figsize=(10, 8))

# 绘制热力图，调整字体大小、颜色映射和数字显示格式
sns.heatmap(survival_rate_by_age_group_pclass, annot=True, fmt=".2f", cmap='coolwarm',
            annot_kws={'size': 8}, cbar_kws={'label': 'Survival Rate'})

# 设置x轴和y轴标签
plt.xlabel('Pclass', fontsize=12)
plt.ylabel('Age Group', fontsize=12)

# 设置标题
plt.title('Survival Rate by Age Group and Pclass', fontsize=14)

# 旋转x轴和y轴的刻度标签
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# 显示图表
plt.tight_layout()  # 自动调整子图参数，使其填充整个图像区域
plt.show()

