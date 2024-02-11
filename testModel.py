import os
import cv2
import sys
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

from emotionClassify import EmotionClassify
from datasetLoader import FERDataset

device = torch.device("mps")
model = EmotionClassify()
model.to(device)

dataset_path, model_path = ('./dataset', sys.argv[1])
classes = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')
face_cascade = cv2.CascadeClassifier('face_cascade.xml')

transformation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])

dataset = FERDataset(csv_file=f'{dataset_path}/val.csv', img_dir=f'{dataset_path}/val/', datatype='finaltest', transform=transformation)
test_loader = DataLoader(dataset, batch_size=64, num_workers=0)

print("Deep Emotion:-", model)
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()

def getImage(path):
    img = Image.open(path)
    img = transformation(img)
    img = img.unsqueeze(0)
    return img.to(device)

cap = cv2.VideoCapture(0)
while True:
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi = img[y:y + h, x:x + w]
        roi = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
        roi = cv2.resize(roi, (48,48))
        cv2.imwrite('tempImage.jpg', roi)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    try:
        img_test = getImage('tempImage.jpg')
    except:
        continue

    out = model(img_test)
    resultClass = F.softmax(out).argmax()
    prediction = classes[int(resultClass.item())]

    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2
    img = cv2.putText(img, prediction, org, font, fontScale, color, thickness, cv2.LINE_AA)

    cv2.imshow('Detection', img)
    k = cv2.waitKey(30)
    if k == ord('q'):
        break

os.remove('tempImage.jpg')
cap.release()
