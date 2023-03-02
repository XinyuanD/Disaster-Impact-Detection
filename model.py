import json
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
# from torchvision.io import read_image
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torch.nn.functional as F
from torchvision.ops import complete_box_iou_loss
from torch.utils.tensorboard import SummaryWriter

def polygon_to_box(polygon):
    str = polygon[10:-2]
    points = str.split(", ")
    xs = []
    ys = []
    for p in points:
        coord = p.split(" ")
        xs.append(float(coord[0]))
        ys.append(float(coord[1]))
    
    xmin = min(xs)
    xmax = max(xs)
    ymin = min(ys)
    ymax = max(ys)
    box = [xmin, ymin, xmax, ymax]

    return box



dmg_intensity = dict()
dmg_intensity["no-damage"] = 1
dmg_intensity["minor-damage"] = 2
dmg_intensity["major-damage"] = 3
dmg_intensity["destroyed"] = 4
dmg_intensity["un-classified"] = 0



class BuildingDamageDataset(Dataset):
    def __init__(self, label_dir, img_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.label_files = os.listdir(self.label_dir)
        print(self.label_files[0])
        self.img_files = os.listdir(self.img_dir)
        self.transform = transform
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    
    def __len__(self):
        return len(self.label_files)
    
    def __getitem__(self, idx):
        with open(os.path.join(self.label_dir, self.label_files[idx])) as f:
            label = json.load(f)["features"]["xy"]
        boxes = [[1e-04, 1e-04, 2 * 1e-04, 2 * 1e-04]] * 64
        labels = [0] * 64
        upperbound = len(label)
        if upperbound > 64:
            upperbound = 64
        
        for i in range(0, upperbound):
            if 'subtype' in label[i]['properties']:
                labels[i] = dmg_intensity[label[i]["properties"]["subtype"]]
            else:
                labels[i] = 1
            boxes[i] = polygon_to_box(label[i]["wkt"])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        image = os.path.join(self.img_dir, self.img_files[idx])
        image = plt.imread(image)
        labels = torch.tensor(labels, dtype=torch.int64)

        if self.transform:
            image = self.transform(image)
            image = image.to(dtype=torch.float32, device=self.device)
        
        return image, boxes, labels



dataset = BuildingDamageDataset("/home/xinyu/train/labels/", "/home/xinyu/train/images/", transform=ToTensor())
train, test = torch.utils.data.random_split(dataset, [4478, 1120])



class Network(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.base = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, padding=3, stride=5, dtype=torch.float32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=7, padding=2, stride=3, dtype=torch.float32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, padding=2, stride=3, dtype=torch.float32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1, dtype=torch.float32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1, dtype=torch.float32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Flatten(),
            nn.Linear(in_features=131072, out_features=1024, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=512, dtype=torch.float32),
            nn.ReLU()
        )

        self.box_final_layer = nn.Linear(in_features=512, out_features=256, dtype=torch.float32)
        self.class_final_layer = nn.Linear(in_features=512, out_features=320, dtype=torch.float32)
    
    def forward(self, x):
        x = self.base(x)

        box_output = self.box_final_layer(x) # 256 -> 64 x 4
        class_output = self.class_final_layer(x) # 320 -> 64 x 5 -> raw numbers

        return box_output, class_output

model = Network().to(torch.device('cuda'))
train_dataloader = DataLoader(train, batch_size=4, shuffle=True)
test_dataloader = DataLoader(test, batch_size=4, shuffle=True)


epochs = 15

train_loss = 0
total_train_steps = 0

optim = torch.optim.Adam(model.parameters(), lr=3e-04)
learning_rate_decay = 0.95
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optim, gamma=learning_rate_decay)
experiment_id = f'lr_3e-04_lr_decay_{learning_rate_decay}_epochs_{epochs}'
writer = SummaryWriter(log_dir=f'log_dir/{experiment_id}')

loss_function = torch.nn.CrossEntropyLoss()



for epoch in range(epochs):
    for images, box_ground_truth, labels_ground_truth in train_dataloader:

        batch_size = images.shape[0]
        if batch_size != 4: continue
            
        output_boxes, output_labels = model.forward(images.to(torch.device('cuda'))) # (x1, y1, w, h)
        output_boxes = output_boxes.T

        try:
            transformed_outputs = torch.stack([output_boxes[:, 0::4], output_boxes[:, 1::4], 
                                               output_boxes[:, 0::4] + output_boxes[:, 2::4] + 1e-03, 
                                        output_boxes[:, 1::4] + output_boxes[:, 3::4] + 1e-03], dim=-1).flatten(start_dim=0, end_dim=1)
        except:
            print(transformed_outputs.shape)
        
        copied_box_ground_truth = box_ground_truth.view((-1, 4)).to(torch.device('cuda')) # this works
        copied_labels_ground_truth = labels_ground_truth.view((batch_size*64)).to(torch.device('cuda'))
        transformed_labels = output_labels.view((batch_size*64, 5))
            
        loss = (complete_box_iou_loss(transformed_outputs, copied_box_ground_truth, reduction='mean', eps=1e-03) + 
                loss_function(transformed_labels, copied_labels_ground_truth))
        train_loss += loss.item()
        total_train_steps += 1
        if torch.isnan(loss):
            print('loss nan')
            break

        optim.zero_grad()
        loss.backward()
        optim.step()

        writer.add_scalar('loss', loss.item(), total_train_steps)
    
    
    lr_scheduler.step()
    print(f'Epoch{epoch} average loss: {round(train_loss/total_train_steps, 3)}')

    eval_ciou_loss = 0
    eval_label_loss = 0
    total_examples_visited = 0

    for images, box_ground_truth, labels_ground_truth in test_dataloader:
        with torch.no_grad():
            batch_size = images.shape[0]
            if batch_size != 4:
                continue

            output_boxes, output_labels = model.forward(images.to(torch.device('cuda'))) # (x1, y1, w, h)
            output_boxes = output_boxes.T
            transformed_outputs = torch.stack([output_boxes[:, 0::4], output_boxes[:, 1::4], output_boxes[:, 0::4] + output_boxes[:, 2::4] + 1e-03, 
                                        output_boxes[:, 1::4] + output_boxes[:, 3::4] + 1e-03], dim=-1).flatten(start_dim=0, end_dim=1)
            copied_box_ground_truth = box_ground_truth.view((-1, 4)).to(torch.device('cuda')) # this works
            copied_labels_ground_truth = labels_ground_truth.view((batch_size*64)).to(torch.device('cuda'))
            transformed_labels = output_labels.view((batch_size*64, 5))

            overlap_loss = complete_box_iou_loss(transformed_outputs, copied_box_ground_truth, reduction='mean', eps=1e-03) 
            label_loss = loss_function(transformed_labels, copied_labels_ground_truth)

            eval_ciou_loss += overlap_loss.item()
            eval_label_loss += label_loss.item()
            total_examples_visited += 1
    
    writer.add_scalar('eval_ciou_loss', eval_ciou_loss / total_examples_visited, total_train_steps)
    writer.add_scalar('eval_label_loss', eval_label_loss / total_examples_visited, total_train_steps)

torch.save(model, f'models/{experiment_id}')

# model = torch.load("models/lr_3e-04_lr_decay_0.95_epochs_10")

all_gt_labels = []
all_pred_labels = []
for images, box_ground_truth, labels_ground_truth in test_dataloader:
    with torch.no_grad():
        batch_size = images.shape[0]
        if batch_size != 4:
            continue

        output_boxes, output_labels = model.forward(images.to(torch.device('cuda'))) # (x1, y1, w, h)
        output_boxes = output_boxes.T
        transformed_outputs = torch.stack([output_boxes[:, 0::4], output_boxes[:, 1::4], 
                                           output_boxes[:, 0::4] + output_boxes[:, 2::4] + 1e-03, 
                                        output_boxes[:, 1::4] + output_boxes[:, 3::4] + 1e-03], dim=-1).flatten(start_dim=0, end_dim=1)
        copied_box_ground_truth = box_ground_truth.view((-1, 4)).to(torch.device('cuda')) # this works
        copied_labels_ground_truth = labels_ground_truth.view((batch_size*64))
        transformed_labels = output_labels.view((batch_size*64, 5))

        pred_labels = torch.argmax(transformed_labels, dim=1)
        all_gt_labels.append(copied_labels_ground_truth.cpu().numpy())
        all_pred_labels.append(pred_labels.cpu().numpy())

accuracy_sum = 0
for i in range(0, len(all_gt_labels)):
    accuracy = (all_pred_labels[i] == all_gt_labels[i]).sum() / len(all_gt_labels[i])
    accuracy_sum += accuracy

accuracy = round(accuracy_sum/len(all_gt_labels), 3) * 100
print(f'Overall accuracy: {accuracy}%')