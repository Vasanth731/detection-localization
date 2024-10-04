import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

class Adapter(nn.Module):

    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x
    

class TextEncoder(nn.Module):

    def __init__(self, classnames, model, device):
        super().__init__()
        self.classnames = classnames
        self.model = model
        self.dtype = model.dtype
        self.prompt = 'an image of a fetal {}.'

    def forward(self):
        prompts = [self.prompt.format(c.replace('_', ' ')) for c in self.classnames]
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(device)
        text_features = self.model.encode_text(prompts)
        x = text_features
        return x


class CustomCLIP(nn.Module):

    def __init__(self, classnames, model, device):
        super().__init__()
        self.image_encoder = model.visual
        self.text_encoder = TextEncoder(classnames, model, device)
        self.logit_scale = model.logit_scale
        self.dtype = model.dtype
        self.adapter = Adapter(512, 4).to(model.dtype)
        self.device = device
        self.ratio = 0.2

            
    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        x = self.adapter(image_features)

        image_features = self.ratio * x + (1 - self.ratio) * image_features
        text_features = self.text_encoder()

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


# if __name__ == "__main__":
#     DATA_DIR = "path_to_the_dataset" 
#     transform = transforms.Compose([
#         preprocess, 
#     ])
#     dataset = ImageFolder(root=DATA_DIR, transform=transform)
#     classnames = dataset.classes
#     custom_clip_model = CustomCLIP(classnames, model, device).to(device)
#     print(f"custom_clip_model : {custom_clip_model}")



