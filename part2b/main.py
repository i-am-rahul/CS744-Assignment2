import torch.distributed as dist
import os
from torchvision import datasets, transforms
from torch.utils.data.distributed import DistributedSampler
import torch
import model as mdl
import torch.distributed as dist
import random
import numpy as np
import argparse


NUM_THREADS = 5
backend = "gloo"
master_ip = "10.10.1.1"
master_port = 29500 + 1
rank = 0
world_size = 1
batch_size = int(256/world_size)
device="cpu"
torch.set_num_threads(NUM_THREADS)
print_every_iteration = 20

def init_process(master_ip, rank, size, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = master_ip
    os.environ['MASTER_PORT'] = str(master_port)
    os.environ['GLOO_SOCKET_IFNAME'] = "eth1"
    dist.init_process_group(backend, rank=rank, world_size=size)
#    vgg_model(rank, size)


def train_model(rank, model, train_loader, optimizer, criterion, epoch=0):
    model.train()
    total_loss = 0
    correct = 0
    group = dist.new_group([_ for _ in range(world_size)])

    for i, (input, target) in enumerate(train_loader):
        input, target = input.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(input)
        train_loss = criterion(output, target)
        train_loss.backward()

        for p in model.parameters():
            dist.all_reduce(p.grad, op=dist.reduce_op.SUM, group=group, async_op=False)
            p.grad /= world_size
        optimizer.step()
        total_loss += train_loss.item()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()


        if i % print_every_iteration == 0:
            print("loss: ", train_loss.item(), "|acc: (", correct, ") ", 100.*correct/len(train_loader.dataset),
                  "%|avgLoss: ", total_loss / (i+1.), "|rank: ", rank)

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


def create_model():
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

    # partition data
    is_distributed = torch.distributed.is_available()
    sampler = DistributedSampler(training_set) if is_distributed else None

    train_loader = torch.utils.data.DataLoader(training_set,
                                                    num_workers=2,
                                                    batch_size=batch_size,
                                                    sampler=sampler,
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
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=0.0001)
    # running training for one epoch
    for epoch in range(1):
        train_model(rank, model, train_loader, optimizer, training_criterion)
        test_model(model, test_loader, training_criterion)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', dest='rank', type=int, help='rank, 0')
    parser.add_argument('--num-nodes', dest='num_nodes', type=int, help='num-nodes, 4')
    args = parser.parse_args()
    global rank,world_size,batch_size
    rank = args.rank
    world_size = args.num_nodes
    batch_size = int(256 / world_size)
    print("running rank = " + str(rank))
    init_process(master_ip, rank, world_size)
    create_model()

if __name__ == "__main__":
    seed = 960904
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    main()