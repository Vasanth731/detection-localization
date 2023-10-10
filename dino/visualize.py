# visualizing predicted class and attention maps
# classic imports
import torch
import torch.nn as nn
from model import dino_classifier
from PIL import Image
from torchvision import transforms as pth_transforms
import matplotlib.pyplot as plt
import numpy as np 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# hyper parameters
input_size = 224 * 224 * 3
image_size =224
PATCH_SIZE = 16 
data_dir = "/home/htic/Pictures/dataset_img"
# data_dir = "/home/endodl/codes_vasanth/dataset_img"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pretrain_path = "/home/htic/Videos/resnet_checkpoint/DINO_best_dino_resnet_classifier_checkpoint.pth"
# pretrain_path = "/home/endodl/codes_vasanth/dino_classifier_edited_4/classifier_checkpoint/DINO_best_dino_classifier_checkpoint.pth"


# setting model 
model = dino_classifier(input_size) # load the pretrained weights 
model.to(device)

model_pretrained_statedict = torch.load(pretrain_path)
model.load_state_dict(model_pretrained_statedict["state_dict"])


transform = pth_transforms.Compose([
    pth_transforms.Resize(image_size),
    pth_transforms.ToTensor(),
    pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
model.eval()


####################### printing images ############################################

# image = Image.open("/home/htic/Pictures/cropped_images/crop_(0, 242)_(0, 242).png")
image = Image.open("/home/htic/Pictures/dataset_img/needle driver/clip_024351.png")
# image = Image.open("/home/htic/Pictures/Screenshot from 2023-08-19 17-24-03.png")
# image = Image.open("/home/htic/Pictures/Screenshot from 2023-08-19 17-26-05.png") # used
# image = Image.open("/home/endodl/codes_vasanth/dataset_img/needle driver/clip_024351.png")
image = image.convert('RGB')
image = transform(image)

# make the image divisible by the patch size
w, h = image.shape[1] - image.shape[1] % PATCH_SIZE, image.shape[2] - image.shape[2] % PATCH_SIZE
image = image[:, :w, :h].unsqueeze(0)

w_featmap = image.shape[-2] // PATCH_SIZE
h_featmap = image.shape[-1] // PATCH_SIZE
attentions = model.student_multicrop.backbone.get_last_selfattention(image.to(device))
# print(attentions.shape) # (1,6,197,197)


nh = attentions.shape[1] # number of head

# we keep only the output patch attention
attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
attentions = attentions.reshape(nh, w_featmap, h_featmap)
attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=PATCH_SIZE, mode="nearest")[0].cpu().detach().numpy()
# print(attentions.shape) # (6,224,224)


################## printing class ########################################

# img_path = "/home/htic/Pictures/cropped_images/crop_(0, 242)_(0, 242).png"
img_path = "/home/htic/Pictures/dataset_img/needle driver/clip_024351.png"
test_image = Image.open(img_path)
test_input = transform(test_image).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(test_input)
    _, predicted_class = torch.max(output, 1)


train_dataset = datasets.ImageFolder(root=data_dir, transform=None)

# Map the predicted class index to the class label
class_labels = train_dataset.classes
predicted_label = class_labels[predicted_class.item()]

print("Original class:", img_path.split('/')[-2])
print("Predicted class:", predicted_label)


# plotting 
plt.figure(figsize=(20,20))
plt.subplot(1,7,1,title='attention_map1'+  '\nPredicted class: ' + predicted_label)
plt.imshow(attentions[1,:,:])
plt.subplot(1,7,2,title='attention_map2')
plt.imshow(attentions[2,:,:])
plt.subplot(1,7,3,title='attention_map3')
plt.imshow(attentions[3,:,:])
plt.subplot(1,7,4,title='attention_map4')
plt.imshow(attentions[4,:,:])
plt.subplot(1,7,5,title='attention_map3')
plt.imshow(attentions[5,:,:])
plt.subplot(1,7,6,title='attention_map4')
plt.imshow(attentions[0,:,:])
# plt.xlabel(f"Predicted class: {predicted_label}\n Original class: {img_path.split('/')[-2]}")
# image=np.swapaxes(image,0,2)

# plt.show()

plt.subplot(1,7,7,title='original_image')
# image=np.swapaxes(image,0,2)
img = np.transpose(image[0],(1,2,0))
plt.imshow(img)
plt.show()