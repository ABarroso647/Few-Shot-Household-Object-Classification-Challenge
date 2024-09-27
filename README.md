The goal for this project was to classify a set of objects contained within a total of 22 classes. The first step was to look at the sample data with an example from class 0 as shown below:
 ![image](https://github.com/user-attachments/assets/87470b79-2b25-4630-9339-83e4efbecb25)
From this overview of the data a couple things could be noted as follows:
•	The images all contained the same grey background for the most part baring some of the lines along the edges
•	The objects were mostly centered in the images
•	There was very minimal training data overall


There are multiple python files which can be ran for different solutions to the issue.
**basic_resnet.py:** This utilizes transfer learning with  ResNet50 model pretrained on the ImageNet Dataset, this was setup as a fixed feature extractor due to low sample data.
This achieves an overall accuracy of 0.84516 on the public score
**prototypical_train.py**: This utilizes the few shot learning technique of prototypical networks to try to address this probem.A prototypical network is based on the idea that there exists an embedding in which point cluster around a single mean or prototype, which represents each class. 
This is done by learning the feature set of objects, and then taking the mean of all the features of the same class to be the class prototype. From there, classification occurs by finding the nearest class prototype. Euclidean distance was used as the distance classified.
From there for this python file, we fine tune the ResNet50 model using the prototypical network.
This achieves an overall accuracy of 0.83032 on the public score (likely lower as the data is low so re-training may add in biases)
**submission.py**: This utilizes the same setup as prototypical_train.py however, in this case we take the pre-trained ResNet50 and simply use the prototypical network as a more advanced classifier on top of it.
This achieves an overall accuracy of 0.89032 on the public score and was the submitted method for the competition.

The overall private score was 0.86740

