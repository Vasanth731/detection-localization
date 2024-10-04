import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import clip
from model import CustomCLIP
# from data_loader import * # uncomment this to use custom dataloader script

# Hyperparameters
BATCH_SIZE = 8
NUM_WORKERS = 4
# MAX_GRAD_NORM = 1.0
NUM_EPOCHS = 100
VAL_SPLIT = 0.2
DATA_DIR = "path_to_dataset"
CKPT_PATH = "ckpt/new_clip_adpt_lastcheckpoint.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model, preprocess = clip.load('ViT-B/32', device)
transform = transforms.Compose([
    preprocess,
])

dataset = ImageFolder(root=DATA_DIR, transform=transform)
train_len = int(0.8 * len(dataset))
val_len = len(dataset) - train_len
train_subset, val_subset = random_split(dataset, [train_len, val_len])

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# # uncomment this to use custom dataloader
# dataset = CustomDataset(root_dir=DATA_DIR, transform=transform)
# classnames = dataset.classes
# train_loader, val_loader = get_dataloaders(DATA_DIR, batch_size=BATCH_SIZE, val_split=VAL_SPLIT, transform=transform)

classnames = dataset.classes
print(classnames)

custom_clip_model = CustomCLIP(classnames, model, device).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(custom_clip_model.parameters(), lr=1e-4)

# Training and validation functions
def train_one_epoch(model, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    loop = tqdm(train_loader)

    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        
        # torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        loop.set_postfix(loss=loss.item())

    accuracy = correct / total
    print(f"Train Epoch [{epoch+1}/{NUM_EPOCHS}] - Loss: {running_loss/len(train_loader):.4f} - Accuracy: {accuracy:.4f}")
    return running_loss / len(train_loader)

def val_one_epoch(model, val_loader, criterion, epoch):
 
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    loop = tqdm(val_loader)

    with torch.no_grad():
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            val_loss += loss.item()

            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            loop.set_postfix(loss=loss.item())

    accuracy = correct / total
    print(f"Val Epoch [{epoch+1}/{NUM_EPOCHS}] - Loss: {val_loss/len(val_loader):.4f} - Accuracy: {accuracy:.4f}")
    return val_loss / len(val_loader)

def save_checkpoint(model, epoch, best=False):
    checkpoint = {'model_state_dict': model.state_dict(), 'epoch': epoch}
    filename = f"{CKPT_PATH if not best else CKPT_PATH.replace('lastcheckpoint', 'bestcheckpoint')}"
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at Epoch {epoch + 1}")


def main():
    best_val_loss = 1e9
    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(custom_clip_model, train_loader, optimizer, criterion, epoch)
        val_loss = val_one_epoch(custom_clip_model, val_loader, criterion, epoch)
        
        save_checkpoint(custom_clip_model, epoch)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(custom_clip_model, epoch, best=True)


if __name__ == "__main__":
    main()
