# 导入所需的库
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from node2vec import Node2Vec
from sklearn.ensemble import RandomForestClassifier

# 加载MAT文件数据
mat_file_path = r"MMM_data.mat"
data = scipy.io.loadmat(mat_file_path)
labels = data['lable'].flatten()

# 定义一个函数来绘制单个图
def plot_individual_graph(adj_matrix, node_labels, compound_index):
    G = nx.from_numpy_array(adj_matrix)
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, node_size=700, with_labels=True, labels={i: str(label[0]) for i, label in enumerate(node_labels)}, node_color='skyblue')
    plt.title(f'Compound {compound_index + 1}')
    plt.show()

# 显示前5个化合物图
num_compounds_to_display = 5
for i in range(min(num_compounds_to_display, len(data['MMM'][0]))):
    compound = data['MMM'][0][i]
    adj_matrix = compound['am']
    node_labels = compound['al']
    plot_individual_graph(adj_matrix, node_labels, i)

# 定义一个函数来提取图的特征
def extract_features(graph):
    G = nx.from_numpy_array(graph)
    features = []
    features.append(nx.density(G))  # 图的密度
    features.append(nx.average_clustering(G))  # 平均集聚系数
    if nx.is_connected(G):
        features.append(nx.average_shortest_path_length(G))  # 平均最短路径长度
    else:
        features.append(-1)  # 如果图不连通，用-1表示

    # 处理度分布直方图，确保长度一致
    degree_hist = nx.degree_histogram(G)
    max_degree = 10  # 假设最大度为10
    degree_hist += [0] * (max_degree + 1 - len(degree_hist))  # 填充零以保证长度一致
    degree_hist = degree_hist[:max_degree + 1]  # 截断确保不超过最大长度
    features.extend(degree_hist)

    return features

# 读取数据
file_path = 'MMM_data.xlsx'
adjacency_matrices = pd.read_excel(file_path, sheet_name='Adjacency Matrices', engine='openpyxl')
node_labels = pd.read_excel(file_path, sheet_name="Node Labels")
graph_labels = pd.read_excel(file_path, sheet_name="Graph Labels")

# 解析节点标签
def parse_node_labels(node_label_str):
    node_label_str = node_label_str.strip('[]')
    labels = [int(item) for item in node_label_str.split()]
    return labels

node_labels_list = node_labels['Node Labels'].apply(parse_node_labels)
graph_labels_array = graph_labels['Graph Labels'].values

# 提取特征向量
def extract_features(node_labels):
    mean_feature = np.mean(node_labels)
    std_feature = np.std(node_labels)
    max_feature = np.max(node_labels)
    min_feature = np.min(node_labels)
    return np.array([mean_feature, std_feature, max_feature, min_feature])

X = np.array([extract_features(labels) for labels in node_labels_list])
y = graph_labels_array

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 使用支持向量机进行分类
svc = SVC(kernel='linear')
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(f'Classification Report: {report}')

# 重复加载MAT文件数据，这可能是代码中的错误，因为之前已经加载过
mat_file_path = r"MMM_data.xlsx"
data = scipy.io.loadmat(mat_file_path)
labels = data['lable'].flatten()

# 解析节点标签
def parse_node_labels(node_label_str):
    node_label_str = node_label_str.strip('[]')
    labels = [int(item) for item in node_label_str.split()]
    return labels

node_labels_list = node_labels['Node Labels'].apply(parse_node_labels)
graph_labels_array = graph_labels['Graph Labels'].values

# 提取特征向量
def extract_features(node_labels):
    mean_feature = np.mean(node_labels)
    std_feature = np.std(node_labels)
    max_feature = np.max(node_labels)
    min_feature = np.min(node_labels)
    return np.array([mean_feature, std_feature, max_feature, min_feature])

X = np.array([extract_features(labels) for labels in node_labels_list])
y = graph_labels_array

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 使用支持向量机进行分类
svc = SVC(kernel='linear')
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(f'Classification Report: {report}')

mat_file_path = r"MMM_data.xlsx"
data = scipy.io.loadmat(mat_file_path)
labels = data['lable'].flatten()

def plot_individual_graph(adj_matrix, node_labels, compound_index):
    G = nx.from_numpy_array(adj_matrix)
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, node_size=700, with_labels=True, labels={i: str(label[0]) for i, label in enumerate(node_labels)}, node_color='skyblue')
    plt.title(f'Compound {compound_index + 1}')
    plt.show()

num_compounds_to_display = 5
for i in range(min(num_compounds_to_display, len(data['MMM'][0]))):
    compound = data['MMM'][0][i]
    adj_matrix = compound['am']
    node_labels = compound['al']
    plot_individual_graph(adj_matrix, node_labels, i)

def extract_features_with_embeddings(graph):
    G = nx.from_numpy_array(graph)
    features = []
    features.append(nx.density(G))
    features.append(nx.average_clustering(G))

    if nx.is_connected(G):
        features.append(nx.average_shortest_path_length(G))
        features.append(nx.diameter(G))
    else:
        features.append(-1)
        features.append(-1)

    node2vec = Node2Vec(G, dimensions=20, walk_length=16, num_walks=100, workers=2)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    embeddings = model.wv.vectors.mean(axis=0)
    features.extend(embeddings.tolist())

    return np.array(features, dtype=float)

features_matrix = []
for compound in data['MMM'][0]:
    adj_matrix = compound['am']
    features = extract_features_with_embeddings(adj_matrix)
    features_matrix.append(features)

features_matrix = np.array(features_matrix)
features_matrix = np.nan_to_num(features_matrix, nan=-1)

X_train, X_test, y_train, y_test = train_test_split(features_matrix, labels, test_size=0.2, random_state=42)
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Classification Accuracy: {accuracy}')
report = classification_report(y_test, y_pred)
print('Classification Report:')
print(report)
