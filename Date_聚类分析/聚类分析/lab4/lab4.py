import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置环境变量以避免内存泄漏
os.environ['OMP_NUM_THREADS'] = '2'

# 读取数据
data = pd.read_csv('data.csv')

# 数据预处理
# 处理缺失值
imputer = SimpleImputer(strategy='mean')
data[['PH', 'IFP', 'NLP', 'NGP', 'NGL', 'NS', 'MHG', 'GY']] = imputer.fit_transform(data[['PH', 'IFP', 'NLP', 'NGP', 'NGL', 'NS', 'MHG', 'GY']])

# 检查异常值并处理（例如，使用Z-score方法）
features = data[['PH', 'IFP', 'NLP', 'NGP', 'NGL', 'NS', 'MHG', 'GY']]
z_scores = np.abs((features - features.mean()) / features.std())
data = data[(z_scores < 3).all(axis=1)]

# 特征工程
# 选择特征列
features = data[['PH', 'IFP', 'NLP', 'NGP', 'NGL', 'NS', 'MHG', 'GY']]

# 相关性分析
correlation_matrix = features.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# PCA降维
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

pca = PCA(n_components=2)  # 保留两个主成分以便可视化
features_pca = pca.fit_transform(features_scaled)

# 可视化PCA降维结果
plt.figure(figsize=(10, 8))
plt.scatter(features_pca[:, 0], features_pca[:, 1])
plt.title('PCA Result')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# 使用肘部法则确定K值
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(features_pca)
    sse.append(kmeans.inertia_)

# 绘制肘部图
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), sse, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('SSE')
plt.show()

# 根据肘部图选择K值，假设我们选择K=3
k = 3

# 创建K-means模型
kmeans = KMeans(n_clusters=k, random_state=42)

# 拟合模型
kmeans.fit(features_pca)

# 获取聚类标签
labels = kmeans.labels_

# 将聚类标签添加到原始数据中
data['Cluster'] = labels

# 可视化聚类结果
plt.figure(figsize=(10, 8))
plt.scatter(features_pca[:, 0], features_pca[:, 1], c=labels, cmap='viridis')
plt.title('K-means Clustering Result')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# 输出结果
print(data.head())

# 可以进一步分析每个簇的特征
for i in range(k):
    cluster_data = data[data['Cluster'] == i]
    print(f"Cluster {i}:\n{cluster_data.describe()}\n")