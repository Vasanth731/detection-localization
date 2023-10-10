# classic imports
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path
from tqdm import tqdm 
import wandb
wandb.login()

user = "vasanth-ambrose"
project = "dino"
display_name = "dino-visual"

wandb.init(entity=user, project=project, name=display_name)

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# imports from own lib
import utils
import vit_model
from img_aug import DataAugmentationDINO
from vit_model import vit_tiny
from vit_model import DINOHead
from loss import DINOLoss
#from dataset import toolsDataset
from vit_model import PatchEmbed

# hyperparameters
learning_rate = 1e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 4
start_epoch = 0
num_epochs = 100
num_workers = 2
beta_1 = 0.5
beta_2 = 0.999
image_height = 224
image_width = 224
pin_memory = True
load_model = True
train_image_dir = "dino_1_AUG/axl_zip/image_dataset/images"
# dino_1_AUG/axl_zip/image_dataset/images
# /home/htic/Documents/dataset/images
val_image_dir = "dino_1_AUG/axl_zip/image_dataset/images"
# /home/htic/Documeet/images/staplernts/datas
# /home/htic/Pictures/archive/tiny-imagenet-200/tiny-imagenet-200/train
pretrain_path = "dino_1_AUG/axl_zip/weights/dino_deitsmall16_pretrain.pth" 
# dino_1_AUG/axl_zip/weights/dino_deitsmall16_pretrain.pth
# /home/htic/Downloads/dino_deitsmall16_pretrain.pth
clip_limit = 2
batch_index = 1
global_crops_scale = (0.4, 1.0)
local_crops_scale = (0.05, 0.4)
local_crops_number= 8
clip_grad = 3.0
#optim = "adamw"
log_step_size = 16
#embed_dim = 192
freeze_last_layer = 1
drop_path_rate = 0.1
PATCH_SIZE = 16  

torch.cuda.empty_cache()
random_seed = 0 # or any of your favorite number 
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
np.random.seed(random_seed)
torch.cuda.is_available()



def train_log(loss,step_size):    
    print(f' loss {loss} step {step_size}')
    wandb.log({"DinoLoss": loss},step=step_size)

    # wandb.log({"step" : step_size,"dino_loss": loss})


def Plot(input_image):
    input_image = np.array(input_image)
    input_image=input_image.cpu().detach().numpy().astype(np.float32)
    #num_images=input_image.shape[0]
    for image,pred,targ in zip(input_image):
        image=np.swapaxes(image,0,2)
        
        plt.figure(figsize=(20,20))
        plt.subplot(1,3,1,title='input')
        plt.imshow(image)
        plt.show()


#  defining train_one_epoch
def train_one_epoch(models, data_loader,
                    optimizer,currentepoch,momentum_schedule,dino_loss):
    
    student, teacher = models[0],models[1]
    # criterian_1= losses[0]
    student_optim,teacher_optim = optimizer[0],optimizer[1]
       
    # Counter
    sample_count = 0.0
    
    # Loss tracker
    mean_LS = 0.0
    mean_LHD = 0.0

    # Set to train
    student.train()
    teacher.train()

    # print(len(data_loader))
    # input("df")
    loop = tqdm(data_loader)
    # Go over each batch of the training set
    for batch_idx, (images,_) in enumerate(loop):
      
        images = [im.cuda(non_blocking=True) for im in images]
        teacher_output = teacher(images[:2])  # only the 2 global views pass through the teacher
        # print(len(images))
        student_output = student(images)
        # print(teacher_output.shape)
        # print(student_output.shape)
        loss = dino_loss(student_output, teacher_output, currentepoch)
        # student update
        student_optim.zero_grad()
        loss.backward()
        param_norms = utils.clip_gradients(student,clip_grad)
        utils.cancel_gradients_last_layer(currentepoch,student,freeze_last_layer)
        student_optim.step()


        #EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[batch_idx] # momentum parameter [batch_idx]
            for param_q, param_k in zip(student.parameters(), teacher.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

# logging losses in wandb 
        if batch_idx%log_step_size==0:
            print(f'EPOCH : {currentepoch}')
            print(f'dino_loss  : {loss}')

            #Plot(loss,(batch_idx*log_step_size)) 
            if currentepoch==0:
                train_log(loss.item(),(batch_idx*log_step_size))
            else:
                train_log(loss.item(),((batch_idx*log_step_size)+(len(data_loader)*log_step_size*currentepoch)))
            
        loop.set_postfix(loss=loss.item())
        
    return [student, teacher], [student_optim,teacher_optim]


def val_one_epoch(models, data_loader,
                    currentepoch,dino_loss):
    
    student, teacher = models[0],models[1]
    # criterian_1= losses[0]
       
    # Counter
    sample_count = 0.0
    
    # Loss tracker
    mean_LS = 0.0
    mean_LHD = 0.0

    # Set to train
    student.eval()
    teacher.eval()

    
    loop = tqdm(data_loader)
    losses = []
    # Go over each batch of the training set
    with torch.no_grad():
        for batch_idx, (images,_) in enumerate(loop):


            images = [im.cuda(non_blocking=True) for im in images]
            teacher_output = teacher(images[:2])  # only the 2 global views pass through the teacher
            student_output = student(images)
            loss = dino_loss(student_output, teacher_output, currentepoch)
            losses.append(loss.item())

            loop.set_postfix(loss=loss.item())
        # mean_LS = mean_LS + loss.item()
        
    return np.mean(losses)


#train loop 
def train_dino():
    # augmenting data
    transform = DataAugmentationDINO(
        global_crops_scale,
        local_crops_scale,
        local_crops_number 
    )
    

    dataset = datasets.ImageFolder(train_image_dir, transform=transform)
    # valtooldataset = toolsDataset(image_dir=val_image_dir, transform=transform) 

    train_data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True, # faster transfer of images from the cpu to the gpu
        drop_last=True,
    )

    
    val_data_loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        num_workers=num_workers,
        pin_memory=True, # faster transfer of images from the cpu to the gpu
        shuffle=False,
    )

    print(f"Data loaded: there are {len(dataset)} images.")


    # declaring student and teacher to vit_tiny architecture
    student_without_multicrop = vit_model.vit_small(drop_path_rate=drop_path_rate,patch_size=PATCH_SIZE)
    teacher = vit_model.vit_small(drop_path_rate=drop_path_rate,patch_size=PATCH_SIZE)
    embed_dim = student_without_multicrop.embed_dim

    #print(embed_dim)

    # loading pretrained weights for student
    student_pretrained_statedict = torch.load(pretrain_path)
    student_without_multicrop.load_state_dict(student_pretrained_statedict) 

    # loading pretained weights for teacher
    teacher_pretrained_statedict = torch.load(pretrain_path)
    teacher.load_state_dict(teacher_pretrained_statedict)

    
    # attaching the DINOHead with the student and teacher
    student = utils.MultiCropWrapper(student_without_multicrop, DINOHead(
        embed_dim,
        out_dim=65536,
        use_bn=False,
        norm_last_layer=True,
    ))
    teacher = utils.MultiCropWrapper(teacher, DINOHead(embed_dim,
        out_dim = 65536,
        use_bn = False),
    )

    
    # moving student and teacher to cuda
    student, teacher = student.cuda(),teacher.cuda()
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False

    # loss function 
    dino_loss = DINOLoss(
        out_dim=65536,
        ncrops=local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        warmup_teacher_temp = 0.04,
        teacher_temp = 0.04,
        warmup_teacher_temp_epochs = 0,
        nepochs=num_epochs,
    ).cuda()

    student_optim = torch.optim.AdamW(student.parameters(),lr=learning_rate, betas=(beta_1,beta_2))
    teacher_optim = torch.optim.AdamW(teacher.parameters(),lr=learning_rate, betas=(beta_1,beta_2))

    # schedulers
    # lr_schedule = utils.cosine_scheduler(
    #     base_value = ((learning_rate) * (batch_size * utils.get_world_size()) / 256.0),  # linear scaling rule
    #     final_value = 1e-6,
    #     epochs = num_epochs, 
    #     niter_per_ep = len(train_data_loader),
    #     warmup_epochs = 10,
    # )
    # wd_schedule = utils.cosine_scheduler(
    #     base_value = 0.04,
    #     final_value = 0.4,
    #     epochs = num_epochs, 
    #     niter_per_ep = len(train_data_loader),
    # )

    momentum_schedule = utils.cosine_scheduler(
        base_value = 0.996, 
        final_value=1,
        epochs=num_epochs, 
        niter_per_ep=len(train_data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    models=[student, teacher]
    optimizers=[student_optim,teacher_optim]

    best_val_loss = 1e9

    for currentepoch in range(start_epoch, num_epochs):
        wandb.watch(student,log="all")

        models, optimizers = train_one_epoch(models,
                                            train_data_loader,
                                            optimizers,
                                            currentepoch, 
                                            momentum_schedule, dino_loss)
        
        val_loss = val_one_epoch(models,
                                val_data_loader,
                                currentepoch, dino_loss)
        
        student_checkpoint = {
                "state_dict": student.state_dict(),
                "optimizer":student_optim.state_dict(),
            }
        model_name="student"

        utils.save_checkpoint(student_checkpoint,currentepoch,model_name)
        
        teacher_checkpoint = {
            "state_dict": teacher.state_dict(),
            "optimizer":teacher_optim.state_dict(),
        }
        model_name="teacher"

        utils.save_checkpoint(teacher_checkpoint,currentepoch,model_name)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            student_checkpoint = {
                "state_dict": student.state_dict(),
                "optimizer":student_optim.state_dict(),
            }
            model_name="best_student"

            utils.save_checkpoint(student_checkpoint,currentepoch,model_name)
            
            teacher_checkpoint = {
                "state_dict": teacher.state_dict(),
                "optimizer":teacher_optim.state_dict(),
            }
            model_name="best_teacher"

            utils.save_checkpoint(teacher_checkpoint,currentepoch,model_name)

        #scheduler.step()


if __name__ == "__main__":
    train_dino()