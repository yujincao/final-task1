from argparse import Namespace
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import GetTransformedDataset
from resnet18 import ResNet_18, ResBlock
import matplotlib.pyplot as plt

def main():
    args = Namespace
    args.batch_size = 1024
    args.device = torch.device('cuda')
    args.epochs = 100
    args.lr = 0.01
    args.weight_decay = 0.0001
    args.workers = 4

    # Load the pre-trained CIFAR10 ResNet18 model
    model = ResNet_18(ResBlock, num_classes=10)
    checkpoint = torch.load('runs/Jun12_09-20-50_zlg76apcejj-e4gbd23cihvy-main/checkpoint_0020.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])

    # Freeze all weights except the last layer
    for param in model.parameters():
        param.requires_grad = False
    model.linear = nn.Linear(model.linear.in_features, 100)
    model.to(args.device)

    # Load the CIFAR100 dataset
    dataset = GetTransformedDataset()
    train_dataset = dataset.get_cifar100_train()
    test_dataset = dataset.get_cifar100_test()
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # Build the optimizer and loss function
    optimizer = optim.SGD(model.linear.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Train the Linear Classification Protocol
    train_acc_list = []
    test_acc_list = []
    train_loss_list = []
    test_loss_list = []

    for epoch in range(args.epochs):
        train_loss = 0.0
        train_correct = 0
        total = 0

        model.train()
        for data in train_loader:
            images, labels = data
            images = images.to(args.device)
            labels = labels.to(args.device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_acc = 100. * train_correct / total
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss / len(train_loader))
        print(f'Epoch [{epoch+1}/{args.epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')

        # Evaluate the performance on CIFAR100 test set
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            test_loss = 0.0
            for data in test_loader:
                images, labels = data
                images = images.to(args.device)
                labels = labels.to(args.device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            test_acc = 100 * correct / total
            test_acc_list.append(test_acc)
            test_loss_list.append(test_loss / len(test_loader))
            print(f'Epoch [{epoch+1}/{args.epochs}], Test Loss: {test_loss/len(test_loader):.4f}, Test Acc: {test_acc:.2f}%')

    # Plot the accuracy and loss graphs
    plt.figure(figsize=(16, 9))
    plt.rcParams.update({'font.size': 14})

    # Accuracy plot
    plt.subplot(2, 1, 1)
    plt.title('CIFAR-100 Accuracy Plot')
    plt.plot(range(len(train_acc_list)), train_acc_list, c='b')
    plt.plot(range(len(test_acc_list)), test_acc_list, c='g')
    plt.legend(['Train Accuracy', 'Test Accuracy'])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)

    # Loss plot
    plt.subplot(2, 1, 2)
    plt.title('CIFAR-100 Loss Plot')
    plt.plot(range(len(train_loss_list)), train_loss_list, c='b')
    plt.plot(range(len(test_loss_list)), test_loss_list, c='g')
    plt.legend(['Train Loss', 'Test Loss'])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    # Save the figure
    plt.savefig('cifar100_CIFAR10Pre_2_accuracy_loss.png', dpi=300)
    # plt.show()

if __name__ == "__main__":
    main()