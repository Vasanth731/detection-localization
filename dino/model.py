import torch
import torch.nn as nn
import vit_model
from vit_model import DINOHead
from utils import MultiCropWrapper
from resnet18 import ResNet18

input_size = 224 * 224 * 3

def freeze(model, freeze: bool):
    for parameter in model.parameters():
        parameter.requires_grad_(not freeze)


class dino_classifier(nn.Module):
    def __init__(self,input):
        super(dino_classifier, self).__init__()
        #hyperparameters 
        DROP_PATH_RATE = 0.1
        PATCH_SIZE = 16
        OUT_DIM = 65536
        new_height, new_width = 224, 224
        pretrain_path_resnet18 = "/home/htic/Videos/resnet18.pth"
        pretrain_path = "/home/htic/Videos/checkpoint/DINO_best_student.pth"

        self.student_pretrained_statedict = torch.load(pretrain_path)
        self.student = vit_model.vit_small(drop_path_rate=DROP_PATH_RATE,patch_size=PATCH_SIZE)
        self.student_multicrop = MultiCropWrapper(self.student, DINOHead(
                                                                    in_dim = self.student.embed_dim,
                                                                    out_dim=OUT_DIM,
                                                                    use_bn=False,
                                                                    norm_last_layer=True,
                                                                    ))
        
        self.student_multicrop.load_state_dict(self.student_pretrained_statedict["state_dict"]) 

        self.conv = nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1)

        self.resnet = ResNet18()
        self.pretrained_state_dict = torch.load(pretrain_path_resnet18)
        self.pretrained_model = self.resnet
        self.pretrained_state_dict = self.pretrained_model.state_dict()
        self.model_state_dict = self.resnet.state_dict()

        # Filter out unnecessary keys and load pretrained weights
        self.pretrained_state_dict = {k: v for k, v in self.pretrained_state_dict.items() if k in self.model_state_dict}
        self.model_state_dict.update(self.pretrained_state_dict)
        self.resnet.load_state_dict(self.model_state_dict)

        # freeze(self.student_multicrop,True)    
        # freeze the layers before this and only update the below layer
        # self.resized_attention_map = torch.nn.functional.interpolate(attention_map, size=(new_height, new_width), mode='bilinear', align_corners=False)
        # self.MLP =MLP(OUT_DIM,14)

        
    def forward(self, input_image):
        attention_map = self.student_multicrop.backbone.get_last_selfattention(input_image)
        resized_attention_map = torch.nn.functional.interpolate(attention_map, size=(224, 224), mode='bilinear', align_corners=False)
        concatenated = torch.concat((input_image,resized_attention_map),dim = 1)
        channel_reduction = self.conv(concatenated)
        x = self.resnet(channel_reduction)
        return x
    
    
    
    
# x = self.student_multicrop(x)
a = torch.rand(1,3,224,224)
if __name__ == "__main__":
    model = dino_classifier(a)
# print(model.forward(a).shape)

# use this to save the model, if any modification is made
# model_path = '/home/endodl/codes_vasanth/model_resnet_with_weights_classifier.pth'
# torch.save(model.state_dict(), model_path)




