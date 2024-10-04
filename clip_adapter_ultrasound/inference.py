import torch
from torchvision import transforms
from PIL import Image
import clip
import torch.nn as nn
import cv2
import numpy as np
from model import *

MODEL_WEIGHTS_PATH = "clip_adp_18_cls_8020split.pth"  
CLASSNAMES = ['4 CHAMBER VIEW', 'ABDOMINAL CIRCUMFERENCE', 'BPD  HC', 'CEREBELLUM', 'CORD INSERTION', 'FACE - PROFILE', 'FEMUR LENGTH', 'KIDNEYS TRANSVERSE', 'LATERAL VENTRICLES & CSP', 'LIQUOR1', 'LVOT', 'ORBITS', 'PLACENTA', 'PMT', 'RVOT', 'SPINE', 'STOMACH - BLADDER', 'THREE  VESSEL VIEW - PAS']

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

transform = transforms.Compose([
    preprocess,
])

def inference(IMAGE_PATH):

    # img_array = np.frombuffer(img_rgb.bits(), dtype=np.uint8).reshape((img_rgb.height(), img_rgb.width(),4))
    # img_array_bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    image = Image.open(IMAGE_PATH)
    image = transform(image).unsqueeze(0).to(device).to(torch.float16)  

    # Load the model
    custom_clip_model = CustomCLIP(CLASSNAMES, model, device).to(device)
    custom_clip_model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH))
    custom_clip_model.eval()

    # Perform inference
    with torch.no_grad():
        logits = custom_clip_model(image)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        top_prob, top_label = probs.topk(1, dim=-1)
        
        predicted_label = CLASSNAMES[top_label.item()]
        confidence = top_prob.item()

        return predicted_label, confidence


if __name__ == "__main__":

    IMAGE_PATH = "path_to_image.jpg"
    predicted_label, confidence = inference(IMAGE_PATH)
    print(predicted_label,confidence)