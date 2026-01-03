import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

# 转换成按批次迭代的对象
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print("Shape of X [N,C,H,W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpui"
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)
# 获取第一个线性层
first_layer = model.linear_relu_stack[0]

print(f"Weight shape: {first_layer.weight.shape}") 
# 输出: torch.Size([512, 784])

print(f"Bias shape: {first_layer.bias.shape}")
# 输出: torch.Size([512])
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # X[64, 1, 28, 28], y[64]
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)
        # 清空梯度:torch默认会累加梯度
        optimizer.zero_grad()
        # 计算梯度
        loss.backward()
        # 更新参数
        optimizer.step()
        

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    # 共多少张图
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    # 开启测试模式
    model.eval()
    
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            # X[64, 1, 28, 28], y[64]
            X, y = X.to(device), y.to(device)
            
            pred = model(X)
            # loss_fun内置:softmax + logloss
            test_loss += loss_fn(pred, y)
            # argmax:获取维度1最大值对应的索引
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {100*correct:0.1f}%, Avg loss: {test_loss:>8f}\n")

epochs = 64
for t in range(epochs):
    print(f"Epoch {t+1}\n---------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

#保存模型
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

#
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth", weights_only=True))

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
model.eval()
X, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    X = X.to(device)
    pred = model(X)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')