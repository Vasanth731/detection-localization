import torch
import numpy as np
import clip
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix 
from model import *

# hyperparameters
MODEL_WEIGHTS_PATH = "path_to_the_saved_weights_path" 
INFERENCE_DIR = "path_to_the_validation_dataset" 

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)
transform = transforms.Compose([
    preprocess, 
])

dataset = ImageFolder(root=INFERENCE_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)
classnames = dataset.classes

custom_clip_model = CustomCLIP(classnames, model, device).to(device)
custom_clip_model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH))
custom_clip_model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(dataloader):
        images = images.to(device).to(torch.float16)
        
        logits = custom_clip_model(images)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        _, top_labels = probs.topk(1, dim=-1)
        
        all_preds.extend(top_labels.squeeze().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

conf_matrix = confusion_matrix(all_labels, all_preds, labels=range(len(classnames)))

# Plot confusion matrix using Seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=classnames, yticklabels=classnames)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png', bbox_inches='tight')
plt.show()
