import torch
from torchvision import datasets, transforms
from autograd import V, L
from model import Model, Ly, O

trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

batch_size = 100 
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

def data_gen(loader):
    for data, target in loader:
        data = data.view(data.shape[0], -1)
        yield V.of(data.numpy()), target.numpy()

model = Model(
    Ly.Sequential(
        Ly.Linear(784, 128),
        Ly.ReLU(),
        Ly.Linear(128, 10),
        Ly.Softmax(),
    ),
    # O.Adam(lr=0.001),
    O.SGD(lr=0.01),
    L.crossentropyloss,
)

model.train(data_gen(train_loader), epoch=10000)
print(model.statistics())
