#!/usr/bin/env python
# coding: utf-8

# # **1. Install packages**

# In[ ]:


# get_ipython().run_cell_magic('capture', '', '!pip install torchinfo\n')


# # **2. Import libraries**

# In[ ]:


# Data handling
import pandas as pd
import numpy as np

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image

# Preprocessing
from sklearn.model_selection import train_test_split as tts

# Torch
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
from torchvision.models import vit_b_16, ViT_B_16_Weights

# Metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report

# os
import os

# OrderedDict
from collections import OrderedDict

# tqdm
from tqdm.auto import tqdm

# Path
from pathlib import Path

# random
import random

# typing
from typing import Dict, List

# warnings
import warnings
warnings.filterwarnings("ignore")

# path
from pathlib import Path

# # **3. Load data and EDA**

# In[ ]:
# Total Images
# IMAGE_PATH = Path("./server/Dataset")

# IMAGE_PATH_LIST = list(IMAGE_PATH.glob("*/*/*.jpg"))

# print(f'Total Images = {len(IMAGE_PATH_LIST)}')

from pathlib import Path

IMAGE_PATH = Path("./Dataset")
print(f"Checking path: {IMAGE_PATH.resolve()}")  

if not IMAGE_PATH.exists():
    print(f"The directory '{IMAGE_PATH}' does not exist.")
else:
    IMAGE_PATH_LIST = list(IMAGE_PATH.glob("**/*.jpg"))
    print(f'Total Images = {len(IMAGE_PATH_LIST)}')



# In[ ]:


# number of images per class.
classes = os.listdir(IMAGE_PATH)
classes = sorted(classes)

print("**" * 20)
print(" " * 10, f"Total Classes = {len(classes)}")
print("**" * 20)

for c in classes:
    total_images_class = list(Path(os.path.join(IMAGE_PATH, c)).glob("*/*.jpg"))
    print(f"* {c}: {len(total_images_class)} images")


# In[ ]:


# We view some images for each class.
NUM_IMAGES = 3

fig, ax = plt.subplots(nrows = len(classes), ncols = NUM_IMAGES, figsize = (10,15))
p = 0
for c in classes:
    total_images_class = list(Path(os.path.join(IMAGE_PATH, c)).glob("*/*.jpg"))
    images_selected = random.choices(total_images_class, k = NUM_IMAGES)
    
    for i,img_path in enumerate(images_selected):
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        ax[p,i].imshow(img_rgb)
        ax[p,i].axis("off")
        ax[p,i].set_title(f"Class: {c}\nShape: {img_rgb.shape}", fontsize = 8, fontweight = "bold", color = "black")
        
    p += 1
    
fig.tight_layout()
fig.show()


# # **4. Preprocessing**

# Let's create a dataframe with two columns:
# 
# - the first **path** call will store the paths of the images.
# - the second call **label** that will contain the labels of each image.

# In[ ]:


images_path = [None] * len(IMAGE_PATH_LIST)
labels = [None] * len(IMAGE_PATH_LIST)

for i,image_path in enumerate(IMAGE_PATH_LIST):
    images_path[i] = image_path
    labels[i] = image_path.parent.parent.stem
    
df_path_and_label = pd.DataFrame({'path':images_path, 
                                  'label':labels})
df_path_and_label.head()


# Now we have to divide our dataframe into 3 parts:
# 
# - **train = 70%**
# - **valid = 15%**
# - **test = 15%**

# In[ ]:


SEED = 123

df_train, df_rest = tts(df_path_and_label, 
                        test_size = 0.3, 
                        random_state = SEED, 
                        stratify = df_path_and_label["label"])

df_val, df_test = tts(df_rest, 
                      test_size = 0.5, 
                      random_state = SEED, 
                      stratify = df_rest["label"])


# In[ ]:


# We have to define the mapping of the classes to convert the labels to numbers.
label_map = dict(zip(classes, range(0, len(classes))))
label_map


# In[ ]:


# Now we define the transformations that we are going to apply.
weights = ViT_B_16_Weights.DEFAULT
auto_transforms = weights.transforms()
auto_transforms


# Now we are going to create the **Dataset's** and **DataLoader's**.
# 
# Let's start with the **Dataset's**.

# In[ ]:


class CustomDataset(Dataset):
    def __init__(self, df:pd.DataFrame, transforms, label_map:dict):
        self.df = df
        self.transforms = transforms
        self.label_map = label_map
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        df_new = self.df.copy()
        df_new = df_new.reset_index(drop = True)
        df_new["label"] = df_new["label"].map(self.label_map)
        image_path = df_new.iloc[idx, 0]
        image = Image.open(image_path).convert("RGB")
        image = self.transforms(image)
        label = df_new.iloc[idx, 1]
        
        return image,label


# In[ ]:


train_dataset = CustomDataset(df_train, auto_transforms, label_map)
valid_dataset = CustomDataset(df_val, auto_transforms, label_map)


# **DataLoader's**

# In[ ]:


BATCH_SIZE = 1
NUM_WORKERS = os.cpu_count()

train_dataloader = DataLoader(dataset = train_dataset, 
                              batch_size = BATCH_SIZE, 
                              shuffle = True, 
                              num_workers = NUM_WORKERS)
valid_dataloader = DataLoader(dataset = valid_dataset, 
                              batch_size = BATCH_SIZE, 
                              shuffle = True, 
                              num_workers = NUM_WORKERS)


# In[ ]:

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)

# Let's visualize the dimensions of a batch.
batch_images, batch_labels = next(iter(train_dataloader))

batch_images.shape, batch_labels.shape


# # **5. Model**

# In[ ]:


# GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
device


# In[ ]:


# We define the model to use with the pre-trained weights.
model = vit_b_16(weights = weights)


# In[ ]:


# Let's visualize the architecture of the model.
summary(model = model, 
        input_size = [1, 3, 224, 224], 
        col_names = ["input_size", "output_size", "num_params", "trainable"], 
        col_width = 15, 
        row_settings = ["var_names"])


# We are going to **freeze the parameters of the conv_proj and encoder layers**.

# In[ ]:


for param in model.conv_proj.parameters():
    param.requires_grad = False


# In[ ]:


for param in model.encoder.parameters():
    param.requires_grad = False


# In[ ]:


# Let's see if the parameters were frozen.
summary(model = model, 
        input_size = [1,3,224,224], 
        col_names = ["input_size", "output_size", "num_params", "trainable"], 
        col_width = 15,
        row_settings = ["var_names"])


# Great!!, the parameters were frozen.
# 
# Let's visualize the last layer which we will modify the number of **out_features**, in this case it is the **number of classes we have**.

# In[ ]:


output_shape = len(classes)

model.heads = nn.Sequential(OrderedDict([('head', nn.Linear(in_features = 768, 
                                                            out_features = output_shape))]))


# In[ ]:


# One last time let's take a look if the last layer was modified.
summary(model = model, 
        input_size = [1,3,224,224], 
        col_names = ["input_size", "output_size", "num_params", "trainable"], 
        col_width = 15,
        row_settings = ["var_names"])


# Let's define the **loss function** and the **optimizer**.

# In[ ]:


loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)


# We are going to define 3 functions for training and one to store the best model:
# 
# - **train_step**
# - **save_checkpoint**
# - **valid_step**
# - **train**

# In[ ]:


def train_step(model:torch.nn.Module, 
               dataloader:torch.utils.data.DataLoader, 
               loss_fn:torch.nn.Module, 
               optimizer:torch.optim.Optimizer):
    
    model.train()
    
    train_loss = 0.
    train_accuracy = 0.
    
    for batch,(X,y) in enumerate(dataloader):
        X,y = X.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred_logit = model(X)
        loss = loss_fn(y_pred_logit, y)
        train_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        
        y_pred_prob = torch.softmax(y_pred_logit, dim = 1)
        y_pred_class = torch.argmax(y_pred_prob, dim = 1)
        train_accuracy += accuracy_score(y.cpu().numpy(), 
                                         y_pred_class.detach().cpu().numpy())
        
    train_loss = train_loss/len(dataloader)
    train_accuracy = train_accuracy/len(dataloader)
    
    return train_loss, train_accuracy


# In[ ]:


def save_checkpoint(filename, model, loss, epoch, optimizer, metric):
    state = {"filename":filename, 
             "model":model.state_dict(), 
             "loss":loss, 
             "epoch":epoch, 
             "optimizer":optimizer.state_dict(), 
             "metric":metric}
    
    torch.save(state, filename)


# In[ ]:


def valid_step(model:torch.nn.Module, 
               dataloader:torch.utils.data.DataLoader, 
               loss_fn:torch.nn.Module):
    
    model.eval()
    
    valid_loss = 0.
    valid_accuracy = 0.
    
    with torch.inference_mode():
        for batch,(X,y) in enumerate(dataloader):
            X,y = X.to(device), y.to(device)
            y_pred_logit = model(X)
            loss = loss_fn(y_pred_logit, y)
            valid_loss += loss.item()
            
            y_pred_prob = torch.softmax(y_pred_logit, dim = 1)
            y_pred_class = torch.argmax(y_pred_prob, dim = 1)
            
            valid_accuracy += accuracy_score(y.cpu().numpy(), y_pred_class.detach().cpu().numpy())
            
    valid_loss = valid_loss/len(dataloader)
    valid_accuracy = valid_accuracy/len(dataloader)
    
    return valid_loss, valid_accuracy


# In[ ]:


def train(model:torch.nn.Module, 
          train_dataloader:torch.utils.data.DataLoader, 
          valid_dataloader:torch.utils.data.DataLoader, 
          loss_fn:torch.nn.Module, 
          optimizer:torch.optim.Optimizer, 
          epochs:int = 10):
    
    results = {"train_loss":[], 
               "train_accuracy":[], 
               "valid_loss":[], 
               "valid_accuracy":[]}
    
    best_valid_loss = float("inf")
    
    for epoch in tqdm(range(epochs)):
        train_loss, train_accuracy = train_step(model = model, 
                                                dataloader = train_dataloader, 
                                                loss_fn = loss_fn, 
                                                optimizer = optimizer)
        
        valid_loss, valid_accuracy = valid_step(model = model, 
                                                dataloader = valid_dataloader, 
                                                loss_fn = loss_fn)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            file_name = "./best_model.pth"
            save_checkpoint(file_name, model, best_valid_loss, epoch, optimizer, valid_accuracy)
            
        print(f"Epoch: {epoch + 1} | ", 
              f"Train Loss: {train_loss:.4f} | ", 
              f"Train Accuracy: {train_accuracy:.4f} | ", 
              f"Valid Loss: {valid_loss:.4f} | ", 
              f"Valid Accuracy: {valid_accuracy:.4f}")
        
        results["train_loss"].append(train_loss)
        results["train_accuracy"].append(train_accuracy)
        results["valid_loss"].append(valid_loss)
        results["valid_accuracy"].append(valid_accuracy)
        
    return results


# In[ ]:


# Training!!!

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=0)

EPOCHS = 100

torch.cuda.manual_seed(SEED)
torch.manual_seed(SEED)


import torch

train_losses = []
valid_losses = []
valid_metrics = []

# Assuming you have a trained model, optimizer, loss function, dataloaders
epochs = 50  # Example number of epochs

def train(model, train_dataloader, valid_dataloader, loss_fn, optimizer, epochs):
    best_loss = float('inf')  # Initialize best loss as infinity
    best_metric = 0.0  # Initialize best metric (like accuracy) as 0 if higher is better
    MODEL_RESULTS = {
        "train_loss": [],
        "valid_loss": [],
        "train_accuracy": [],
        "valid_accuracy": []
    }

    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        epoch_train_loss = 0.0
        model.train()
        correct_train = 0
        total_train = 0
        
        # Training loop
        for batch in train_dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()  # Clear gradients
            outputs = model(inputs)  # Forward pass
            loss = loss_fn(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update model weights
            
            loss = loss_fn(outputs, labels)
            epoch_train_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
            
            epoch_train_loss += loss.item()  # Sum training loss
        
        epoch_train_loss /= len(train_dataloader)  # Average training loss
        train_losses.append(epoch_train_loss)  # Store train loss for this epoch
        train_accuracy = correct_train / total_train
        
        # Validation loop
        model.eval()  # Set the model to evaluation mode
        epoch_valid_loss = 0.0
        epoch_valid_metric = 0.0
        correct_valid = 0
        total_valid = 0
        
        with torch.no_grad():  # No need to calculate gradients during validation
            for batch in valid_dataloader:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)  # Forward pass
                loss = loss_fn(outputs, labels)  # Compute loss
                epoch_valid_loss += loss.item()  # Sum validation loss
                
                # Calculate accuracy/metric (optional, depends on your task)
                # Example: accuracy = (predicted_labels == actual_labels).sum().item() / len(labels)
                # validation_metric += accuracy
                
                loss = loss_fn(outputs, labels)
                epoch_valid_loss += loss.item()
                preds = outputs.argmax(dim=1)
                correct_valid += (preds == labels).sum().item()
                total_valid += labels.size(0)
        
        epoch_valid_loss /= len(valid_dataloader)  # Average validation loss
        valid_losses.append(epoch_valid_loss)  # Store validation loss
        valid_accuracy = correct_valid / total_valid
        
        # Optionally store validation metric
        valid_metrics.append(epoch_valid_metric)
        
        MODEL_RESULTS["train_loss"].append(epoch_train_loss)
        MODEL_RESULTS["valid_loss"].append(epoch_valid_loss)
        MODEL_RESULTS["train_accuracy"].append(train_accuracy)
        MODEL_RESULTS["valid_accuracy"].append(valid_accuracy)
        
        # Save model if validation loss improves
        if epoch_valid_loss < best_loss:
            best_loss = epoch_valid_loss
            checkpoint = {
                'epoch': epoch,  # Save the current epoch
                'model_state_dict': model.state_dict(),  # Save model weights
                'optimizer_state_dict': optimizer.state_dict(),  # Save optimizer state
                'loss': best_loss,  # Save the best loss
                'metric': epoch_valid_metric  # Optionally save the best metric
            }
            torch.save(checkpoint, 'best_model.pth')  # Save best model weights
            print(f"Best model saved at epoch {epoch} with loss {best_loss}")
        
        # Optionally save model if metric improves
        if epoch_valid_metric > best_metric:
            best_metric = epoch_valid_metric
            torch.save(model.state_dict(), 'best_model_metric.pth')  # Save best metric model
            print(f"Best model saved at epoch {epoch} with metric {best_metric}")
            
    return MODEL_RESULTS


# In[ ]:


# Function to plot the loss and metric during each training epoch.
def loss_metric_curve_plot(model_results:Dict[str,List[float]]):
    
    train_loss = model_results["train_loss"]
    valid_loss = model_results["valid_loss"]
    
    train_accuracy = [float(value) for value in model_results["train_accuracy"]]
    valid_accuracy = [float(value) for value in model_results["valid_accuracy"]]
    
    fig,axes = plt.subplots(nrows = 1, ncols = 2, figsize = (10,4))
    axes = axes.flat
    
    axes[0].plot(train_loss, color = "red", label = "Train")
    axes[0].plot(valid_loss, color = "blue", label = "Valid")
    axes[0].set_title("CrossEntropyLoss", fontsize = 12, fontweight = "bold", color = "black")
    axes[0].set_xlabel("Epochs", fontsize = 10, fontweight = "bold", color = "black")
    axes[0].set_ylabel("Loss", fontsize = 10, fontweight = "bold", color = "black")
    axes[0].legend()
    
    axes[1].plot(train_accuracy, color = "red", label = "Train")
    axes[1].plot(valid_accuracy, color = "blue", label = "Valid")
    axes[1].set_title("Metric of performance: Accuracy", fontsize = 12, fontweight = "bold", color = "black")
    axes[1].set_xlabel("Epochs", fontsize = 10, fontweight = "bold", color = "black")
    axes[1].set_ylabel("Score", fontsize = 10, fontweight = "bold", color = "black")
    axes[1].legend()
    
    fig.tight_layout()
    fig.show()


# In[ ]:

MODEL_RESULTS = train(model.to(device), 
                      train_dataloader, 
                      valid_dataloader, 
                      loss_fn, 
                      optimizer, 
                      EPOCHS)

print(f'Model Results: {MODEL_RESULTS}')
loss_metric_curve_plot(MODEL_RESULTS)


# In[ ]:


# Let's load the best model.
checkpoint_path = "./best_model.pth"
checkpoint = torch.load(checkpoint_path)
print(checkpoint)

# In[ ]:


# Now let's look at the smallest loss, its metric and when it occurred.
print(f'Best Loss: {checkpoint["loss"]}')
print(f'Epoch: {checkpoint["epoch"] + 1}')
print(f'Best Metric: {checkpoint["metric"]}')


# Well, now we have to predict the images of the test set.

# - **Predictions**

# In[ ]:


# First of all, we create the Dataset, DataLoader
test_dataset = CustomDataset(df_test, auto_transforms, label_map)
test_dataloader = DataLoader(dataset = test_dataset, shuffle = False, num_workers = NUM_WORKERS)


# In[ ]:


test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

# We define the model again with its respective modification.
loaded_model = vit_b_16()

loaded_model.heads = nn.Sequential(OrderedDict([('head',nn.Linear(in_features = 768, 
                                                                  out_features = output_shape))]))

loaded_model.load_state_dict(checkpoint["model_state_dict"])

# We now infer
loaded_model.to(device)

loaded_model.eval()

y_pred_test = []

with torch.inference_mode():
    for X,y in tqdm(test_dataloader):
        X,y = X.to(device), y.to(device)
        y_pred_logit = loaded_model(X)
        y_pred_prob = torch.softmax(y_pred_logit, dim = 1)
        y_pred_class = torch.argmax(y_pred_prob, dim = 1)
        y_pred_test.append(y_pred_class.detach().cpu())


# In[ ]:


y_pred_test = torch.cat(y_pred_test).numpy()


# # **6. Metrics**

# - **Accuracy**

# In[ ]:


print(f'Accuracy = {round(accuracy_score(df_test["label"].map(label_map), y_pred_test), 4)}')


# - **Confusion Matrix**

# In[ ]:


confusion_matrix_test = confusion_matrix(df_test["label"].map(label_map), y_pred_test)


# In[ ]:


fig,ax = plt.subplots(figsize = (15,4))
sns.heatmap(confusion_matrix_test, 
            cmap = 'coolwarm', 
            annot = True, 
            annot_kws = {"fontsize":9, "fontweight":"bold"}, 
            linewidths = 1.2, 
            linecolor = "black", 
            square = True, 
            xticklabels = classes, 
            yticklabels = classes, 
            cbar = False,
            ax = ax)
ax.set_title("Confusion Matrix Test", fontsize = 10, fontweight = "bold", color = "darkblue")
ax.tick_params('x',rotation = 90)
fig.show()


# %%

# training.py
# import torch

# def load_model(model_path):
#     model = torch.load(model_path)
#     model.eval()  # Set the model to evaluation mode
#     return model

# %%

import torch
from torchvision import transforms
from torchvision.models import vit_b_16  # Import the Vision Transformer
from PIL import Image

# Load the model architecture and weights
model = vit_b_16(weights=None)  # Initialize the ViT model
model.load_state_dict(torch.load('best_model.pth'), strict=False)  # Load the trained weights
model.eval()  # Set the model to evaluation mode

# Define the transformation for input images
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Adjust this based on your model's expected input size
    transforms.ToTensor(),
    # Add any other necessary transformations, like normalization
])

def IAmPredicting(image_path):
    image = Image.open(image_path)
    image = preprocess(image)
    image = image.unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)  # Get the class with the highest score
    
    return predicted.item()  # Return the predicted class index

# %%


# %%
