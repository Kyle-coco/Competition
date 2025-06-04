import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv
from sklearn.model_selection import train_test_split

# 导入数据
file_path = 'MMM_data.xlsx'
adjacency_matrices = pd.read_excel(file_path, sheet_name='Adjacency Matrices')
node_labels = pd.read_excel(file_path, sheet_name="Node Labels")
graph_labels = pd.read_excel(file_path, sheet_name='Graph Labels')

# 检查数据内容
print("Adjacency Matrices:")
print(adjacency_matrices.head())

print("Node Labels:")
print(node_labels.head())

print("Graph Labels:")
print(graph_labels.head())

# 解析邻接矩阵
def parse_adjacency_matrix(matrix):
    if isinstance(matrix, int):
        return np.zeros((1, 1))  # 根据实际情况调整
    else:
        raise ValueError("Unsupported data format for adjacency matrix")

def parse_node_labels(node_label_str):
    if isinstance(node_label_str,str):
        node_label_str=node_label_str.strip('[]')
        labels = list(map(int,node_label_str.split()))
        return labels
    elif isinstance(node_label_str,list):
        return node_label_str
    else:
        raise ValueError("Unsupported data format for adjacency matrix")
adjacency_matrices_list = adjacency_matrices['Adjacency Matrix'].apply(parse_adjacency_matrix)
node_labels_list=node_labels['Node Labels'].apply(parse_node_labels)
node_labels_array=node_labels_list.tolist()
graph_labels_array=graph_labels['Graph Labels'].values
#验证标签数据并修正无效标签
print("Graph Labels unique Values Before Fix:",np.unique(graph_labels_array))
graph_labels_array=np.where(graph_labels_array==-1,0,graph_labels_array)
print("Graph Labels unique Values After Fix:",np.unique(graph_labels_array))
# 创建图数据
data_list = []
for i in range(len(graph_labels_array)):
    adj_matrix=adjacency_matrices_list.iloc[i]
    edge_index = torch.tensor(np.vstack(np.where(adj_matrix)).astype(np.int64),dtype=torch.long)
    x=torch.tensor(node_labels_array[i],dtype=torch.float).unsqueeze(1)
    y = torch.tensor([graph_labels_array[i]],dtype=torch.long)
    data=Data(x=x,edge_index=edge_index,y=y)
    data_list.append(data)
# 划分数据集
train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)
train_loader=DataLoader(train_data, batch_size=1, shuffle=True)
test_loader=DataLoader(test_data,batch_size=1,shuffle=False)

class GAT(torch.nn.Module):
    def __init__(self):
        super(GAT, self).__init__()
        self.convl = GATConv(1, 64, heads=8, concat=True)
        self.conv2 = GATConv(64 * 8, 2, heads=1, concat=True)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.convl(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x,edge_index)
        x = torch.nn.functional.log_softmax(x,dim=1)
        x = torch.mean(x,dim=0)
        return x.unsqueeze(0)


#模型训练与评估
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 使用GPU或CPU
model=GAT().to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=0.01,weight_decay=5e-4)
def train():
    model.train()
    for data in train_loader:
        data=data.to(device)
        optimizer.zero_grad()
        out = model(data)
        #打印模型输出和标签以进行调试
        print("Model Output:",out)
        print("Target:",data.y)
        loss = torch.nn.functional.nll_loss(out,data.y)
        print("Loss:",loss.item())
        loss.backward()
        optimizer.step()
def test(loader):
    model.eval()
    correct=0
    for data in loader:
        data=data.to(device)
        out = model(data)
        pred=out.argmax(dim=1)
        correct +=pred.eq(data.y).sum().item()
    return correct /len(loader.dataset)


for eporch in range(20):
    train()
    train_acc = test(test_loader)
    test_acc = test(test_loader)
    print(f'Eporch:{eporch+1},Train Acc:{train_acc:.4f},TestAcc:{test_acc:.4f}')




