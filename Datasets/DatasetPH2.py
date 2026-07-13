# Archivo para crear Dataset
import os
import torch
import numpy as np
from torchvision import  transforms
from torch.utils.data import Dataset
from PIL import Image, ImageFilter, ImageEnhance

##############################################################################
#                            PH2                                             #
##############################################################################

class PH2_Loader(Dataset):
    def __init__(self, img_path, mask_path, indices, transform=False, conjunto = "train"):
        self.img_path = img_path
        self.mask_path = mask_path
        self.conjunto = conjunto

        new_size = (256,192)
        self.new_size = new_size
        self.consider_resize = True

        # Indices de las imagenes
        self.ids = indices
        self.transform = transform

    def __len__(self):
        return len(self.ids)
    
    def rotate(self, image, mask, degrees=(-15,15), p=0.5):
        if torch.rand(1).item() < p:
            degree = float(torch.empty(1).uniform_(*degrees))
            image = image.rotate(degree, Image.NEAREST)
            mask = mask.rotate(degree, Image.NEAREST)
        return image, mask
    
    def horizontal_flip(self, image, mask, p=0.5):
        if torch.rand(1).item() < p:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return image, mask
    
    def vertical_flip(self, image, mask, p=0.5):
        if torch.rand(1).item() < p:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        return image, mask
    
    def random_resized_crop(self, image, mask, p=0.5, min_mask_fraction=0.01):
        if torch.rand(1).item() < p:
            W, H = image.size

            scale = float(torch.empty(1).uniform_(0.8, 0.95))
            ratio = float(torch.empty(1).uniform_(3/4, 4/3))

            area = H * W * scale
            h = int(round(np.sqrt(area / ratio)))
            w = int(round(np.sqrt(area * ratio)))

            h = min(h, H)
            w = min(w, W)

            i = torch.randint(0, H - h + 1, (1,)).item()
            j = torch.randint(0, W - w + 1, (1,)).item()

            candidate_mask = transforms.functional.resized_crop(
                mask,i,j,h,w,(192, 256),interpolation=transforms.InterpolationMode.NEAREST)
            
            mask_arr = np.asarray(candidate_mask)
            # Only crop if its a min_mask_fraction
            if mask_arr.sum() > mask_arr.size * min_mask_fraction:
                image = transforms.functional.resized_crop(image, i, j, h, w, (192, 256))
                mask = candidate_mask
                
        return image, mask
    
    def gaussian_blur(self, image, p=0.2, radius=(0.5, 1.5)):
        if torch.rand(1).item() < p:
            r = float(torch.empty(1).uniform_(*radius))
            image = image.filter(ImageFilter.GaussianBlur(radius=r))
        return image
    
    def random_brightness(self, image, p=0.3, factor=(0.7, 1.3)):
        if torch.rand(1).item() < p:
            f = float(torch.empty(1).uniform_(*factor))
            image = ImageEnhance.Brightness(image).enhance(f)
        return image
    
    def augment(self, image, mask):
        image, mask = self.random_resized_crop(image, mask)
        image, mask = self.rotate(image, mask)
        image, mask = self.horizontal_flip(image, mask)
        image, mask = self.vertical_flip(image, mask)
        image = self.gaussian_blur(image)
        image = self.random_brightness(image)
        return image, mask

    def __getitem__(self, idx):
        id_ = self.ids[idx]
        img = Image.open(os.path.join(self.img_path, id_ + ".png")).convert("RGB")

        # Train + Validation
        if self.conjunto in ["train", "validation"]:
            mask = Image.open(os.path.join(self.mask_path, id_ + "_lesion.png")).convert("L")
            if self.consider_resize == True:
                img = img.resize(self.new_size)
                mask = mask.resize(self.new_size, Image.NEAREST)

            # Apply data aug
            if self.transform and self.conjunto == "train":
                img, mask = self.augment(img, mask)
        # Test
        else:
            mask = None
            if self.consider_resize == True:
                img = img.resize(self.new_size)

        img = np.asarray(img, dtype=np.float32) / 255.0
        img = torch.from_numpy(img).permute(2,0,1)

        if mask is not None:
            # [0,1] 
            mask = np.asarray(mask, dtype=np.int64)
            mask = (mask > 0).astype(np.int64)
            mask = torch.from_numpy(mask).long()

        # Diccionario
        return {"image": img,"mask": mask, "name": id_}

