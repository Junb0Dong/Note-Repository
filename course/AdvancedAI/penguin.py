from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# 加载数据
train_data = pd.read_csv('train_data.csv')

# 处理特征与目标
X = train_data.drop(columns=['Species'])
y = train_data['Species']

# 对类别特征进行 One-Hot 编码
X = pd.get_dummies(X, drop_first=True)

# 训练随机森林模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 获取特征重要性
importance = model.feature_importances_

# 输出每个特征的重要性
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importance})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
print(feature_importance.head(6))