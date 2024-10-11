import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 1. 定义模型
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 2. 创建数据集
# 假设我们有一些训练数据和标签
train_data = torch.randn(100, 10)  # 100个样本，每个样本10个特征
train_labels = torch.randint(0, 2, (100,))  # 100个样本的标签，二分类问题

# 创建DataLoader
train_dataset = TensorDataset(train_data, train_labels)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

# 3. 初始化模型、损失函数和优化器
input_size = 10
hidden_size = 5
output_size = 2
model = SimpleNN(input_size, hidden_size, output_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 4. 训练模型
num_epochs = 5
for epoch in range(num_epochs):
    for data, labels in train_loader:
        # 前向传播
        outputs = model(data)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 5. 评估模型
# 假设我们有一些测试数据和标签
test_data = torch.randn(20, 10)
test_labels = torch.randint(0, 2, (20,))

# 前向传播
with torch.no_grad():
    test_outputs = model(test_data)
    _, predicted = torch.max(test_outputs.data, 1)
    accuracy = (predicted == test_labels).sum().item() / test_labels.size(0)

print(f'Accuracy: {accuracy * 100:.2f}%')