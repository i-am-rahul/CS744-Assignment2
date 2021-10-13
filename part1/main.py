import torch
from torchvision import datasets, transforms
import torch.optim as optim
import argparse

from datetime import datetime, date, time, timedelta
import model as mdl
import random
import numpy as np

device = "cpu"
NUM_THREADS = 4
torch.set_num_threads(NUM_THREADS)

batch_size = 256 # batch for one node
print_every_iteration = 20
limited_iterations = 40 # -1 for no limit

def train_model(model, train_loader, optimizer, criterion, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch input loader): Training input loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """

    model.train()
    total_loss = 0
    correct = 0
    now = datetime.now()
    iters = 0
    for i, (input, target) in enumerate(train_loader):
        input, target = input.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(input)
        train_loss = criterion(output, target)
        train_loss.backward()
        optimizer.step()

        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        total_loss += train_loss.item()
        iters += 1
        if i == 0:
            # discarding time for 1st round
            now = datetime.now()
            iters = 0
        elif limited_iterations != -1 and i >= limited_iterations - 1:
            break
        if i % print_every_iteration == 0:
            print("loss: ", train_loss.item(), "|acc: (", correct, ") ", 100.*correct/len(train_loader.dataset),
                  "%|avgLoss: ", total_loss / (i+1.), "|i: ", i)
    print("avg training time for ", iters, " -> ", (datetime.now() - now).total_seconds()/iters, "s")
    return None

def test_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            input, target = input.to(device), target.to(device)
            output = model(input)
            test_loss += criterion(output, target)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
            

def main():
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])

    transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize])
    training_set = datasets.CIFAR10(root="./input", train=True,
                                                download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(training_set,
                                                    num_workers=2,
                                                    batch_size=batch_size,
                                                    sampler=None,
                                                    shuffle=False,
                                                    pin_memory=True)
    test_set = datasets.CIFAR10(root="./input", train=False,
                                download=True, transform=transform_test)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              num_workers=2,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True)
    training_criterion = torch.nn.CrossEntropyLoss().to(device)

    model = mdl.VGG11()
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=0.0001)
    # running training for one epoch
    for epoch in range(1):
        train_model(model, train_loader, optimizer, training_criterion, epoch)
        test_model(model, test_loader, training_criterion)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', dest='rank', type=int, help='rank, 0', nargs='?')
    parser.add_argument('--num-nodes', dest='num_nodes', type=int, help='num-nodes, 4', nargs='?')
    parser.add_argument('--seed', dest='seed', type=int, help='seed, starting number', default=1267)
    parser.add_argument('--master-ip', dest='master_ip', type=str, help='seed, starting number', nargs='?')
    args = parser.parse_args()
    global master_ip, seed
    master_ip = args.master_ip
    seed = args.seed
    print("running with seed:", seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

#    np.random.seed(seed)
    main()
