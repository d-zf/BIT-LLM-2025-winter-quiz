import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 设置全局字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体（Windows系统自带）
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 加载嵌入数据和标签
embeddings = np.load('sequence_embeddings.npy')  # shape: (518, 768)

# 数据标准化
scaler = StandardScaler()
embeddings_scaled = scaler.fit_transform(embeddings)

# # Using the elbow method to find the optimal number of clusters
# wcss = []
# for i in range(5, 30):
#     kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
#     kmeans.fit(embeddings_scaled)
#     wcss.append(kmeans.inertia_)

# # 绘制WCSS随簇数量变化的折线图
# plt.plot(range(5, 30), wcss)
# plt.title('The Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')
# plt.savefig(
#     'The Elbow Method.png',
#     dpi=300,
#     bbox_inches='tight',
#     transparent=False
# )
# plt.show()


labels = KMeans(21,random_state=0).fit_predict(embeddings_scaled)

#t-SNE
tsne = TSNE(
    n_components=2,
    perplexity=30,          
    n_iter=1500,
    random_state=42,
    init='pca',
    learning_rate=200,
    verbose=1  # 显示进度信息
)

print("开始t-SNE降维...")
embeddings_2d = tsne.fit_transform(embeddings_scaled)
print("降维完成")

# 可视化设置
plt.figure(figsize=(14, 10))
ax = plt.gca()

plt.scatter(embeddings_2d[:,0],embeddings_2d[:,1], c=labels, s=30,alpha=0.6,cmap='viridis')

plt.title('质粒嵌入空间t-SNE可视化 (n=518)', fontsize=14)
plt.xlabel('t-SNE 维度1', fontsize=12)
plt.ylabel('t-SNE 维度2', fontsize=12)

# 添加辅助网格
ax.grid(True, linestyle='--', alpha=0.4)

# 保存高分辨率图片
plt.savefig(
    'colored_tsne(21 clusters).png',
    dpi=300,
    bbox_inches='tight',
    transparent=False
)

plt.show()