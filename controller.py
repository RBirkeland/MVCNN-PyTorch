import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader
from torch.autograd import Variable

from custom_dataset import MultiViewDataSet

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import numpy as np
import time

import matplotlib.pyplot as plt
from resnet import *

import os


print('Loading data')

transform = transforms.Compose([
    transforms.CenterCrop(500),
    transforms.ToTensor(),
])


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dset_train = MultiViewDataSet('data/train', transform=transform)
train_loader = DataLoader(dset_train, batch_size=4, shuffle=True, num_workers=2)

dset_val = MultiViewDataSet('data/val', transform=transform)
val_loader = DataLoader(dset_val, batch_size=4, shuffle=True, num_workers=2)

dset_test = MultiViewDataSet('data/test', transform=transform)
test_loader = DataLoader(dset_test, batch_size=1, shuffle=True, num_workers=2)

classes = dset_train.classes
print(classes)

resnet = resnet18()
resnet.to(device)
cudnn.benchmark = True

print(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
lr = 0.01
optimizer = torch.optim.Adam(resnet.parameters(), lr=lr)
n_epochs = 10

best_acc = 0
start_epoch = 0


# Helper functions
def load_checkpoint():
    global best_acc, start_epoch
    # Load checkpoint.
    print('\n==> Loading checkpoint..')
    assert os.path.isfile('checkpoint/checkpoint.pth.tar'), 'Error: no checkpoint file found!'

    checkpoint = torch.load('checkpoint/checkpoint.pth.tar')
    best_acc = checkpoint['best_acc']
    start_epoch = checkpoint['epoch']
    resnet.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

def train():
    train_size = len(train_loader)

    for i, (inputs, targets) in enumerate(train_loader):
        # Convert from list of 3D to 4D
        inputs = np.stack(inputs, axis=1)

        inputs = torch.from_numpy(inputs)

        inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)

        # compute output
        outputs = resnet(inputs)
        loss = criterion(outputs, targets)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 5 == 0:
            print("\tIter [%d/%d] Loss: %.4f" % (i + 1, train_size, loss.item()))

def eval():
    # Eval
    total = 0
    correct = 0

    total_loss = 0
    n = 0

    for i, (inputs, targets) in enumerate(val_loader):
        with torch.no_grad():
            # Convert from list of 3D to 4D
            inputs = np.stack(inputs, axis=1)

            inputs = torch.from_numpy(inputs)

            inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)

            # compute output
            outputs = resnet(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss
            n += 1

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted.cpu() == targets.cpu()).sum()

    avg_test_acc = 100 * correct / total
    avg_loss = total_loss / n

    return avg_test_acc, avg_loss

def test():
    load_checkpoint()

    # Eval
    total = 0
    correct = 0

    total_loss = 0
    n = 0

    for i, (inputs, targets) in enumerate(test_loader):
        with torch.no_grad():
            # Convert from list of 3D to 4D
            inputs = np.stack(inputs, axis=1)

            inputs = torch.from_numpy(inputs)

            inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)

            # compute output
            outputs = resnet(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss
            n += 1

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted.cpu() == targets.cpu()).sum()

    avg_test_acc = 100 * correct / total
    avg_loss = total_loss / n

    return avg_test_acc, avg_loss

def save_checkpoint(state, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)


# Training / Eval loop
resume = True

if resume:
    load_checkpoint()

# Training
for epoch in range(start_epoch, n_epochs):
    print('\n-----------------------------------')
    print('Epoch: [%d/%d]' % (epoch+1, n_epochs))
    start = time.time()

    resnet.train()
    train()
    print('Time taken: %d sec.' % (time.time() - start))

    resnet.eval()
    avg_test_acc, avg_loss = eval()

    print('\nEvaluation:')
    print('\tVal Acc: %d - Loss: %.4f' % (avg_test_acc.item(), avg_loss.item()))
    print('\tCurrent best model: %d' % best_acc)

    # Save model
    if avg_test_acc > best_acc:
        print('\tSaving checkpoint - Acc: %d' % avg_test_acc)
        best_acc = avg_test_acc
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': resnet.state_dict(),
            'acc': avg_test_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        })

    # Decaying Learning Rate
    if (epoch + 1) % 20 == 0:
        lr *= 0.99
        optimizer = torch.optim.Adam(resnet.parameters(), lr=lr)
        print('Learning rate:', lr)


resnet.eval()
avg_test_acc, avg_loss = test()

print('\nTest:')
print('\tTest Acc: %d - Loss: %.4f' % (avg_test_acc.item(), avg_loss.item()))
print('\tCurrent best model: %d' % best_acc)
