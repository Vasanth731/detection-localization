import os
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data import WeightedRandomSampler


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = []
        
        for cls in self.classes:
            class_dir = os.path.join(root_dir, cls)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.images.append((img_path, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('RGB')  # Convert to RGB
        if self.transform:
            image = self.transform(image)
        return image, label


def get_dataloaders(root_dir, batch_size=16, val_split=0.2, transform=None):

    dataset = CustomDataset(root_dir=root_dir, transform=transform)

    total_size = len(dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # implementing weighted random sampler
    class_counts = np.zeros(len(dataset.classes))
    for _, label in train_dataset:
        class_counts[label] += 1
    
    sample_weights = [1 / class_counts[label] for _, label in train_dataset]
    train_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_dataset), replacement=True)
    
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader


# # to check the dataloader uncomment the below script
# if __name__ == "__main__":

#     transform = transforms.Compose([
#         transforms.Resize((800, 800)),  
#         transforms.ToTensor(),          
#     ])

#     root_dir = 'dataset2'
#     batch_size = 16
#     val_split = 0.2

#     dataset = CustomDataset(root_dir=root_dir, transform=transform)
#     print(dataset.classes)

#     train_loader, val_loader = get_dataloaders(root_dir, batch_size=batch_size, val_split=val_split, transform=transform)
#     print(train_loader)

#     for images, labels in train_loader:
#         print(images.shape)  
#         print(labels)
#         break  
