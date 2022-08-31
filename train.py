import torch
from torch.utils.data import DataLoader
from VGG16addGAP import *
import Dataset_CIFAR10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

PATH = '../../dataset/cifar-10-batches-py'
# PATH = '../dataset/cifar-10-batches-py'

batch_size = 64

train_dataset = Dataset_CIFAR10.CIFAR10Dataset('train', PATH)
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

test_dataset = Dataset_CIFAR10.CIFAR10Dataset('test', PATH)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)

model = totalNet()
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00003)

def train(epoch):
    running_loss = 0.0
    for batch_index, data in enumerate(train_loader, 0):
        input, targets = data
        input, targets = input.to(device), targets.to(device)

        output = model(input)
        loss = criterion(output, targets)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_index%300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_index + 1, running_loss / 300))
            running_loss = 0.0

def test(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            data, label = data
            data, label = data.to(device), label.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, dim = 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
        print('accuracy on test set: %d %% \n' % (100 * correct / total))

if __name__ == "__main__":
    print("使用"+str(device)+"训练！")
    test(model, test_loader)
    for epoch in range(15):
        train(epoch)
        test(model, test_loader)
    save_model_path = 'state_dict_model.pth'
    torch.save(model.state_dict(), save_model_path)