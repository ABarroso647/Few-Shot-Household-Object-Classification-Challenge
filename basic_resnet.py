import os
import uuid
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader
import numpy as np
from graph import graph_function
from PIL import Image
import natsort
import pandas as pd

# grab the internal device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TrainingData(torchvision.datasets.ImageFolder):
    # updated to associate the folder names with the correct classes
    def find_classes(self, directory: str) -> tuple[list[str], dict[str, int]]:
        class_value = {}
        for i in range(0, 22):
            class_value[str(i)] = i

        classes = class_value.keys()
        classes_to_idx = class_value
        return classes, classes_to_idx


def gen_training_data(transforms, root='data/train'):
    dataset = TrainingData(root=root, transform=transforms)
    return dataset


def train(model, trainloader, optimizer, criterion):
    model.train()
    total_loss = []
    for i, data in enumerate(trainloader):
        images, labels = data

        # uncomment the below 3 lines to view the transformed images
        # warning it will spam
        # transform = transforms.ToPILImage()
        # img = transform(images[0])
        # img.show()
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        total_loss.append(loss.item())
        optimizer.step()

    # return the average losses over teh training batches
    return np.average(total_loss)


def eval_train(model, testloader, criterion):
    model.eval()

    total_loss = []
    correct = 0.0
    predictions = []
    for data in testloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss.append(loss.item())
        # grab the max indicy, convert it to a float and sum all the correct ones
        correct += ((torch.max(outputs, 1).indices == labels).float().sum()).item()
        predictions.append(torch.max(outputs, 1).indices)

    return np.average(total_loss), correct, predictions


class TestDataSet(torch.utils.data.Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        # get a list of all the images from the main directory
        all_imgs = os.listdir(main_dir)

        # put all the images in order
        self.total_imgs = natsort.natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        # get the specific image location
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])

        # convert it to RGB
        image = Image.open(img_loc).convert("RGB")

        # perform the transformations to the image
        tensor_image = self.transform(image)
        return tensor_image, img_loc


if __name__ == '__main__':
    batch_size = 32
    class_num = 22
    train_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # imagenet transform values
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # imagenet transform values
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_datatset = TestDataSet("data/test", transform=test_transforms)
    train_dataset = gen_training_data(train_transforms)
    # shuffle to introduce some randomness
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # eval sets so batch size is pretty large to evaluate it
    train_test = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset))
    test_loader = torch.utils.data.DataLoader(test_datatset, batch_size=len(test_datatset),
                                              shuffle=False)

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # freeze all cnn layers
    for param in model.parameters():
        param.requires_grad = False

    #grab the number of input features of the last layer
    num_ftrs = model.fc.in_features
    #define a new linear layer which takes the in features and classifies to one of the 22 classes
    model.fc = nn.Linear(num_ftrs, class_num)

    lr = 0.01
    weight_decay = 10e-4
    max_epochs = 30
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    model = model.to(device)
    train_loss = []
    train_accur = []
    train_predictions = []
    print('Training losses: ')
    for epoch in range(max_epochs):
        train_loss.append(train(model, train_loader, optimizer, criterion))

        # evaluate how well its actually learning the test data
        _, correct_count, predictions = eval_train(model, train_test, criterion)
        train_accur.append(correct_count / len(train_dataset) * 100.)
        train_predictions.append(predictions)
        print(train_loss[epoch])

    graph_function(train_loss, "Loss", "Train loss vs Epochs")
    graph_function(train_accur, "Accuracy (%)", "Train accuracy vs Epochs")
    print('Training done, running on test data:')
    # evaluate on the test data, and append to a pandas dataframe
    model.eval()
    test_df = pd.DataFrame(columns=['category'])
    for idx, img in enumerate(test_loader):
        images, _ = img
        images = images.to(device)
        outputs = model(images)
        test_predicts = torch.max(outputs, 1).indices.cpu().tolist()
        predict_df = pd.DataFrame(test_predicts, columns=['category'])
        test_df = pd.concat([test_df, predict_df], ignore_index=True)

    # generate a unique filename
    unique_filename = str(uuid.uuid4())
    path = 'submission/' + unique_filename + '.csv'
    # append to a csv, with the appropriate headers
    test_df.to_csv(path, header=True, index_label='id')
    print('Complete :)')

