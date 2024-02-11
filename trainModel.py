import os
import warnings

import torch
import torchvision
import torch.nn as nn

import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from datasetLoader import FERDataset
from emotionClassify import EmotionClassify

# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = '1'
# export PYTORCH_ENABLE_MPS_FALLBACK=1

warnings.filterwarnings("ignore")

device = torch.device("mps")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def TrainModel(epochs, train_loader, val_loader, criterion, optmizer, device):
    for e in range(epochs):
        train_loss, validation_loss, train_correct, val_correct = (0, 0, 0, 0)

        model.train()
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optmizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs,labels)
            loss.backward()
            optmizer.step()
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data)

        model.eval()
        for data,labels in val_loader:
            data, labels = data.to(device), labels.to(device)
            val_outputs = model(data)
            val_loss = criterion(val_outputs, labels)
            validation_loss += val_loss.item()
            _, val_preds = torch.max(val_outputs, 1)
            val_correct += torch.sum(val_preds == labels.data)

        train_loss = train_loss / len(train_dataset)
        train_acc = train_correct / len(train_dataset)

        validation_loss =  validation_loss / len(validation_dataset)
        val_acc = val_correct / len(validation_dataset)
        print(f'[ Epoch: {e+1} \tTraining Loss: {train_loss:.8f} \tValidation Loss {validation_loss:.8f} \tTraining Acuuarcy {train_acc * 100:.3f}% \tValidation Acuuarcy {val_acc*100:.3f}% ]')

    torch.save(model.state_dict(), f'eLU{epochs}e.pt')


if __name__ == '__main__':
    epochs = 150
    lr = 0.005
    batchsize = 128

    model = EmotionClassify()
    model.to(device)
    data = './dataset'

    print("Model archticture: ", model)
    train_csv_file = f'{data}/train.csv'
    val_csv_file = f'{data}/val.csv'
    train_img_dir = f'{data}/train/'
    val_img_dir = f'{data}/val/'

    transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])

    train_dataset = FERDataset(csv_file=train_csv_file, img_dir=train_img_dir, datatype='train', transform=transformation)
    validation_dataset = FERDataset(csv_file=val_csv_file, img_dir=val_img_dir, datatype='val', transform=transformation)
    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=0)
    val_loader = DataLoader(validation_dataset, batch_size=batchsize, shuffle=True, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    optmizer = optim.Adam(model.parameters(), lr=lr)
    TrainModel(epochs, train_loader, val_loader, criterion, optmizer, device)
