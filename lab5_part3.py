import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split


# Check if we can use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


#loading and normalizing the dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "fruit_images")

transform = transforms.Compose(
    [   
        transforms.Resize((64,64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    
    ]
)

dataset = ImageFolder(root=data_path, transform=transform)

#Stacking into tensors
labels = []
images = []
for i in range(len(dataset)):
    img, label = dataset[i]
    labels.append(label)
    images.append(img)

labels = torch.tensor(labels)
images = torch.stack(images)



#Making the stratisized split
x_train, x_test, y_train, y_test = train_test_split(
    images.numpy(),
    labels.numpy(),
    test_size = 0.2,
    stratify = labels.numpy(),
    random_state = 42
)


#Converting x & y test and train back into tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

#creating the test and train data sets
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

#Creating Data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Defining the CNN architecture
class CNN(nn.Module):
    def __init__(self, N, M, num_classes=5, input_shape=(3,64,64), kernal_size = 3):
        super(CNN, self).__init__()

        # making a module list for the convolution layers so we can have a different amount of them
        self.convs = nn.ModuleList()
        in_channels = input_shape[0]

        #Here we dynamically create the ammount of convolution layers we want for the NN
        for i in N:
            conv_layer = nn.Conv2d(in_channels, i, kernel_size=kernal_size, padding=kernal_size//2)
            self.convs.append(conv_layer)
            in_channels = i
        
        self.pool = nn.MaxPool2d(2,2)
        self.dropout = nn.Dropout(0.25)

        with torch.no_grad():
            x = torch.zeros(1, *input_shape)
            for conv in self.convs:
                x = self.pool(F.relu(conv(x)))
            self.flattened_size = x.view(1,-1).size(1)
        
        self.fcs = nn.ModuleList()
        in_features = self.flattened_size
        for out_features in M:
            self.fcs.append(nn.Linear(in_features, out_features))
            in_features = out_features

        self.output = nn.Linear(in_features, num_classes)
    def forward(self, x):
        for conv in self.convs:
            x = self.pool(F.relu(conv(x)))
        
        x = x.view(x.size(0), -1)

        for fc in self.fcs:
            x = self.dropout(F.relu(fc(x)))
        
        x = self.dropout(x)
        return x
model = CNN(N=[16,32,64], M=[500])

print(model)

#For the optimizer and lossfunction
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=5)