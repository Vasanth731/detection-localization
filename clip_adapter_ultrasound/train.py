import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm
import clip
from model import *


# hyperparameters
BATCH_SIZE = 16
NUM_WORKERS = 4
MAX_GRAD_NORM = 1.0
NUM_EPOCHS = 10
DATA_DIR = "path_to_the_train_dataset"
CKPT_PATH = "path_to_save_the_weights.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)
transform = transforms.Compose([
    preprocess, 
])

dataset = ImageFolder(root=DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
classnames = dataset.classes

custom_clip_model = CustomCLIP(classnames, model, device).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(custom_clip_model.parameters(), lr=1e-4) # don't ever give ADAM optimizer here !!!!!

for epoch in range(NUM_EPOCHS):
    custom_clip_model.train()
    running_loss = 0.0
    for images, labels in tqdm(dataloader):
        images, labels = images.to(torch.float16).to(device), labels.to(device) # .to(torch.float16)
        
        logits = custom_clip_model(images) # .to(torch.float16)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        
        # # uncomment to clip the gradients
        # torch.nn.utils.clip_grad_norm_(custom_clip_model.adapter.parameters(), MAX_GRAD_NORM)
        
        optimizer.step()
        
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(dataloader)
    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {epoch_loss:.4f}')

print('Training finished.')

# Save the weights
weights_path = CKPT_PATH
torch.save(custom_clip_model.state_dict(), weights_path)

