from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    analysis_target = [0 for i in range(10)]
    analysis_output = [0 for i in range(10)]
    count = 0
    true_count = 0
    false_count = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            for idx, value in enumerate(target):
                analysis_target[value] += 1
                result = torch.argmax(output[idx])
                if value == result:
                    analysis_output[value] += 1
                    if(true_count < 5):
                        if(true_count == 0):
                            print('\nTrue inference sample images')
                        print('  Result: target "{0}" -> output "{1}"\tfile_path: ./images/true_sample/{2}.jpg'.format(
                            value,result,1000*count+idx))
                        true_count += 1
                else:
                    if(false_count < 5):
                        if(false_count == 0):
                            print('\nFalse inference sample images')
                        print('  Result: target "{0}" -> output "{1}"\tfile_path: ./images/false_sample/{2}.jpg'.format(
                            value,result,1000*count+idx))
                        false_count += 1

            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            count += 1

    test_loss /= len(test_loader.dataset)
    print('\nNumber result')
    for idx, value in enumerate(analysis_target):
        acc = (analysis_output[idx] / value) * 100.0
        print('  Number of {0} images: {1} EA\tAccuracy: {2} %'.format(
            idx, value, round(acc,2)))
    print('\nTotal result')
    print('  Number of total images: {0} EA'.format(len(test_loader.dataset)))
    print('  Average loss: {:.4f}\t\tAccuracy: {:.2f} %\n'.format(
        test_loss, 100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('./data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('./data', train=False,
                       transform=transform)
    
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net()
    model.load_state_dict(torch.load("model_cnn.pt"))
    model.to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    test(model, device, test_loader)

if __name__ == '__main__':
    main()
