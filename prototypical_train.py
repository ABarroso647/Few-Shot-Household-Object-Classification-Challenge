import os
import uuid
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from easyfsl.samplers import TaskSampler
from torchvision import models
from torch.utils.data import DataLoader
from graph import graph_function
from PIL import Image
import natsort
import pandas as pd
from tqdm import tqdm
from easyfsl.utils import sliding_average

# grab the internal device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_WAY = 22  # Number of classes in a task
N_SHOT = 3  # Number of images per class in the support set
N_QUERY = 2  # Number of images per class in the query set
N_TRAINING_EPISODES = 15

class PrototypicalNetwork(nn.Module):
    def __init__(self, backbone: nn.Module):
        super(PrototypicalNetwork, self).__init__()
        self.backbone = backbone
    def forward(
        self,
        support_images,
        support_labels,
        query_images,
    ):
        # Get the support and test features from the base CNN
        z_support = self.backbone.forward(support_images)
        z_query = self.backbone.forward(query_images)

        # Get the number of different classes from the support class
        n_way = len(torch.unique(support_labels))
        # Get the mean of all prototypes corresponding to each labelled class
        z_proto = torch.cat(
            [
                z_support[torch.nonzero(support_labels == label)].mean(0)
                for label in range(n_way)
            ]
        )

        # Get the mean euclidean distance from each prototype to the test data
        dists = torch.cdist(z_query, z_proto)

        # Get the scores from those distances (lowest distance is the best score)
        scores = -dists
        return scores

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


def fit(
        support_images, support_labels, query_images, query_labels, model, optimizer, criterion
):
    # pretty standard training function
    # except we pass all 3 image sets into the model
    optimizer.zero_grad()
    classification_scores = model(
        support_images.to(device), support_labels.to(device), query_images.to(device)
    )
    loss = criterion(classification_scores, query_labels.to(device))
    loss.backward()
    optimizer.step()
    return loss.item()

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

    # needed for tasksampler class, generates list of images labels like 0,0,0,0,1...
    train_dataset.get_labels = lambda: [instance[1] for instance in train_dataset.imgs]

    # creates a task sampler, with our hyperparamaeters
    train_sampler = TaskSampler(
        train_dataset, n_way=N_WAY, n_shot=N_SHOT, n_query=N_QUERY, n_tasks=N_TRAINING_EPISODES
    )

    # passes the task sampler into the dataloader so its enumerate will perform like we expect
    # from a few shot learnign episode
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_sampler, pin_memory=True,
                                               collate_fn=train_sampler.episodic_collate_fn,)

    # can sample the train_test normally, need batch size to be all for evaluation
    train_test = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset.imgs))

    # this batch size is smaller to stop gpu from running out of memory :(
    test_loader = torch.utils.data.DataLoader(test_datatset, batch_size=batch_size,
                                              shuffle=False)

    cnn = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # change the last layer to just be a flatten
    cnn.fc = nn.Flatten()
    model = PrototypicalNetwork(cnn).to(device)


    lr = 0.0001
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # update param for the progress bar
    log_update_frequency = 10
    all_loss = []
    model.train()

    # run through each episode
    # each episode is only one generation of training images
    print('Training: ')
    with tqdm(enumerate(train_loader), total=len(train_loader)) as tqdm_train:
        for episode_index, (
                support_images,
                support_labels,
                query_images,
                query_labels,
                _,
        ) in tqdm_train:
            loss_value = fit(support_images, support_labels, query_images, query_labels, model, optimizer, criterion)
            all_loss.append(loss_value)

            if episode_index % log_update_frequency == 0:
                tqdm_train.set_postfix(loss=sliding_average(all_loss, log_update_frequency))


    print('Training done, running on test data:')
    graph_function(all_loss, "Loss (%)", "Train Loss vs Epochs")


    model.eval()
    test_df = pd.DataFrame(columns=['category'])

    # this is a for loop, but it should only be one loop
    # as epoch size is set to the whole dataset
    for _, data in enumerate(train_test):
        # grab the images from training data to be used for eval
        images, labels = data

        # loop through the test images
        with tqdm(enumerate(test_loader), total=len(test_loader)) as test_classification:
            for idx, test_images in test_classification:

                # grab the test images, these will be the query images
                test_images, _ = test_images
                # pass the sample images into the model along with the query images
                outputs = model(images.to(device), labels.to(device), test_images.to(device)).detach()
                # get the model predicts
                test_predicts = torch.max(outputs, 1).indices.cpu().tolist()
                predict_df = pd.DataFrame(test_predicts, columns = ['category'])
                test_df = pd.concat([test_df, predict_df], ignore_index=True)

    # generate unique file name
    unique_filename = str(uuid.uuid4())
    path = 'submission/' + unique_filename + '.csv'

    # save to csv with proper headers
    test_df.to_csv(path, header=True, index_label='id')
    print('Complete :)')
