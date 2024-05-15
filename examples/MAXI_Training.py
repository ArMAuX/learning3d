# Import statements
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from learning3d.losses import ClassificationLoss
from learning3d.models import PointNet, Classifier
from learning3d.data_utils import ModelNet40Data

# Model initialization
ptnet = PointNet(emb_dims=1024, input_shape='bnc', use_bn=True)
model = Classifier(feature_model=ptnet)


## set modelnet40 = ModelNet40Data(train= True, download= True)
# Dataset initialization
modelnet40 = ModelNet40Data(train=True, num_points=1024)

## trainset ^ 

trainloader = DataLoader(modelnet40, batch_size= 32, shuffle= True, num_workers= 4)


## params missing
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr)


## Defining the epochs
max_epochs = 300
for i in range(max_epochs):
    for i, data in enumerate(tqdm(trainloader)):
        points, target = data
        target = target.squeeze(-1)
        output = model(points)
        loss = ClassificationLoss()
        optimizer.zero_grad()
        optimizer.step()
