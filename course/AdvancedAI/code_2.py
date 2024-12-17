import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def generate_two_classes_data(n_samples = 300):
    X = []
    y = []
    for _ in range(n_samples):
        x = np.random.uniform(-10, 10)
        # 生成第一类数据（y > 2x + 1）
        if np.random.rand() > 0.5:
            y_value = 2 * x + 1 + np.random.uniform(0.1, 10)
            X.append([x, y_value])
            y.append(1)
        # 生成第二类数据（y < 2x + 1）
        else:
            y_value = 2 * x + 1 - np.random.uniform(0.1, 10)
            X.append([x, y_value])
            y.append(0)
    X = np.array(X)
    y = np.array(y)
    return X, y

# 假设这里已经有了生成环形数据集的函数generate_ring_dataset
def generate_ring_dataset(n_samples=150, inner_radius=2, outer_radius=4):
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
X, y = generate_two_classes_data()

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

print(X_train.shape)

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

# 绘制决策树决策边界和数据集
plt.subplot(1, 2, 1)
# 绘制决策边界，使用实线
plt.contour(xx, yy, Z_dt, colors='k', linewidths=1.5, linestyles='-')
# 绘制数据点，用'+'和'-'表示不同类别
for i in range(len(X_train)):
    if i < X_train.shape[0]:  # 增加这个判断条件确保索引合法
        if y_train[i] == 1:
            plt.scatter(X_train[i, 0], X_train[i, 1], c='k', marker='+', s=30)
        else:
            plt.scatter(X_train[i, 0], X_train[i, 1], c='k', marker='o', s=30)
plt.title("Decision Tree")
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.axis('equal')

# 绘制线性模型决策边界和数据集
plt.subplot(1, 2, 2)
# 绘制决策边界，使用虚线
plt.contour(xx, yy, Z_lm, colors='k', linewidths=1.5, linestyles='--')
# 绘制数据点，用'+'和'-'表示不同类别
for i in range(len(X_train)):
    if y_train[i] == 1:
        plt.scatter(X_train[i, 0], X_train[i, 1], c='k', marker='+', s=30)
    else:
        plt.scatter(X_train[i, 0], X_train[i, 1], c='k', marker='o', s=30)
plt.title("Linear Model")
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.axis('equal')

plt.show()