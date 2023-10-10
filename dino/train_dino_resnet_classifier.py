# classic imports
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from datetime import datetime 
from torch.utils.data import DataLoader,Subset,random_split
# from lion_pytorch import Lion

# import from own library 
from model import dino_classifier

# wandb login 
import wandb
wandb.login()
user = "vasanth-ambrose"
project = "dino_classifier"
display_name = "resnet_visual"
wandb.init(entity=user, project=project, name=display_name)

# hyperparameters
data_dir = "/home/endodl/codes_vasanth/dataset_img"
# data_dir = "/home/htic/Pictures/dataset_img"
batch_size = 48
NUM_WORKERS = 2
input_size = 224 * 224 * 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 100
log_step_size = 16
start_epoch = 0
pretrain_path = '/home/endodl/codes_vasanth/model_resnet_with_weights_classifier.pth' # use the weights which is saved in the model.py
# pretrain_path = "/home/endodl/codes_vasanth/vit_multicrop_mlp_joined_with_weights_till_multicrop.pth"

# few needed functions 
def train_log(loss,step_size):    
    print(f' loss {loss} step {step_size}')
    wandb.log({"Loss": loss},step=step_size)

def save_classifier_checkpoint(state,epoch,model_name):
    date=datetime.date(datetime.now())
    time=datetime.time(datetime.now())
    date_time=str(date)+str("__")+str(time)
    date_time=date_time[0:20]

    # /home/htic/Music/dino_classifier_edited_4/classifier_checkpoint
#     filename = f'/home/htic/Music/dino_classifier_edited_4/classifier_checkpoint/DINO_{model_name}.pth'
    filename = f'/home/endodl/codes_vasanth/dino_trial_1/resnet_checkpoint/DINO_{model_name}.pth'

    print("=> Saving checkpoint")
    torch.save(state, filename)
    
def freeze(model, freeze: bool):
    for parameter in model.parameters():
        parameter.requires_grad_(not freeze)   


def train_one_epoch(model,train_loader,optimizer,criterion,currentepoch):

    model = model
    optimizer = optimizer

    model.train()
    correct = 0
    total = 0
    
    loop = tqdm(train_loader)

    for batch_idx, (images,labels) in enumerate(loop):
        images, labels = images.to(device), labels.to(device)
        
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if batch_idx%log_step_size==0:
            print(f'EPOCH : {currentepoch}')
            print(f'loss  : {loss}')

            # Plot(loss,(batch_idx*log_step_size)) 
            if currentepoch==0:
                train_log(loss.item(),(batch_idx*log_step_size))
            else:
                train_log(loss.item(),((batch_idx*log_step_size)+(len(train_loader)*log_step_size*currentepoch)))
            
        loop.set_postfix(loss=loss.item())
        
    # epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct / total
    print(f"Epoch [{currentepoch+1}/{num_epochs}] - Loss: {loss:.4f} - Accuracy: {epoch_accuracy:.4f}")

    return model, optimizer


def val_one_epoch(model,validation_loader,criterion,currentepoch):

    model = model
    model.eval()
    val_correct = 0
    val_total = 0

    loop = tqdm(validation_loader)
    losses = []
    with torch.no_grad():
        for batch_idx, (val_images,val_labels) in enumerate(loop):
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            val_outputs = model(val_images)
            
            loss = criterion(val_outputs,val_labels) 
            _, val_predicted = val_outputs.max(1)
            
            val_total += val_labels.size(0)
            val_correct += val_predicted.eq(val_labels).sum().item()
            losses.append(loss.item())
            loop.set_postfix(loss=loss.item())

        val_accuracy = val_correct / val_total
        print(f"Epoch [{currentepoch+1}/{num_epochs}] - Val Loss: {loss:.4f} - Val Accuracy: {val_accuracy:.4f}")
        mean_loss = np.mean(losses)
        return mean_loss


def train():
    # Data preprocessing and augmentation
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Adjust color
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = datasets.ImageFolder(root=data_dir, transform=data_transform)
    num_classes = len(train_dataset.classes)

    validation_ratio = 0.2  # jus for checking i've given this value it should actually be 0.2
    num_train = len(train_dataset)
    num_validation = int(validation_ratio * num_train)
    num_train = num_train - num_validation
    train_subset, validation_subset = random_split(train_dataset, [num_train, num_validation])


    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)

    train_loader = DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    validation_loader = DataLoader(dataset=validation_subset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)


    model = dino_classifier(input_size)
    # model = normal_mlp(input_size,num_classes)
    
    dino_classifier_pretrained_statedict = torch.load(pretrain_path)
    model.load_state_dict(dino_classifier_pretrained_statedict) 
    model.to(device)
    freeze(model.student_multicrop,True)
    
   

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
#     optimizer = Lion(model.parameters(), lr=1e-4, weight_decay=1e-2)

    best_val_loss = 1e9
    for currentepoch in range(start_epoch, num_epochs):
        wandb.watch(model,log="all")

        model, optimizer = train_one_epoch(model,train_loader,optimizer,criterion,currentepoch)
        mean_loss = val_one_epoch(model,validation_loader,criterion,currentepoch)

        dino_classifier_checkpoint = {
                "state_dict": model.state_dict(),
#                 "optimizer":optimizer.state_dict(),
            }
        model_name="dino_resnet_classifier_checkpoint"
        save_classifier_checkpoint(dino_classifier_checkpoint,currentepoch,model_name)

        if mean_loss < best_val_loss:
            best_val_loss = mean_loss

            dino_classifier_checkpoint = {
                "state_dict": model.state_dict(),
#                 "optimizer":optimizer.state_dict(),
            }
            model_name="best_dino_resnet_classifier_checkpoint"

            save_classifier_checkpoint(dino_classifier_checkpoint,currentepoch,model_name)


    return mean_loss

if __name__ == "__main__":
    train()
