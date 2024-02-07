import os
import uuid
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader
from PIL import Image
import natsort
import pandas as pd
from tqdm import tqdm

# grab the internal device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

        # load and convert it to RGB
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

    # can sample the train_test normally, need batch size to be all for evaluation
    # otherwise performance will decrease
    train_test = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset.imgs))
    test_loader = torch.utils.data.DataLoader(test_datatset, batch_size=batch_size,
                                              shuffle=False)

    cnn = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    cnn.fc = nn.Flatten()
    model = PrototypicalNetwork(cnn).to(device)

    print('Running eval on test data:')
    model.eval()
    test_df = pd.DataFrame(columns=['category'])
    for _, data in enumerate(train_test):
        # grab the images from training data to be used for eval
        images, labels = data
        with tqdm(enumerate(test_loader), total=len(test_loader)) as test_classification:
            for idx, test_images in test_classification:
                # grab the test images, these will be the query images
                test_images, _ = test_images
                # pass the sample images into the model along with the query images
                # technically this is inefficient but it only affects eval time
                # as the prototypes can be saved as static values in one shot and then used
                # instead of running them through the model each time
                outputs = model(images.to(device), labels.to(device), test_images.to(device)).detach()
                # get the model predicts
                test_predicts = torch.max(outputs, 1).indices.cpu().tolist()
                predict_df = pd.DataFrame(test_predicts, columns=['category'])
                test_df = pd.concat([test_df, predict_df], ignore_index=True)

    # generate unique file name
    unique_filename = str(uuid.uuid4())
    path = 'submission/' + unique_filename + '.csv'
    # save to csv with proper headers
    test_df.to_csv(path, header=True, index_label='id')
    print('Complete :)')
