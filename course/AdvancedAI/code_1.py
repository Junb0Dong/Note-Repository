import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 生成圆形数据集
def generate_ring_dataset(n_samples = 300, inner_radius = 2, outer_radius = 4):
    # 生成角度
    theta = 2 * np.pi * np.random.rand(n_samples)
    # 生成内圆半径，最大值小于外圆半径最小值
    inner_r = inner_radius * np.ones(n_samples)
    # 生成外圆半径，最小值大于内圆半径最大值
    outer_r = outer_radius * np.ones(n_samples)
    # 转换为直角坐标
    x_inner = inner_r * np.cos(theta)
    y_inner = inner_r * np.sin(theta)
    x_outer = outer_r * np.cos(theta)
    y_outer = outer_r * np.sin(theta)
    X = np.vstack((np.column_stack((x_inner, y_inner)), np.column_stack((x_outer, y_outer))))
    y = np.hstack((np.ones(len(x_inner)), np.zeros(len(x_outer))))
    return X, y

# 生成数据集
X, y = generate_ring_dataset()

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练决策树模型并评估准确性
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print("Decision Tree Accuracy:", accuracy_dt)

# 训练线性模型（逻辑回归）并评估准确性
lm = LogisticRegression()
lm.fit(X_train, y_train)
y_pred_lm = lm.predict(X_test)
accuracy_lm = accuracy_score(y_test, y_pred_lm)
print("Linear Model Accuracy:", accuracy_lm)

# 绘制决策边界和数据集可视化
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z_dt = dt.predict(np.c_[xx.ravel(), yy.ravel()])
Z_dt = Z_dt.reshape(xx.shape)

Z_lm = lm.predict(np.c_[xx.ravel(), yy.ravel()])
Z_lm = Z_lm.reshape(xx.shape)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.contourf(xx, yy, Z_dt, alpha=0.4)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolor='k')
plt.title("Decision Tree Decision Boundary")

plt.subplot(1, 2, 2)
plt.contourf(xx, yy, Z_lm, alpha=0.4)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolor='k')
plt.title("Linear Model Decision Boundary")
plt.show()