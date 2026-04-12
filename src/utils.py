from PIL import Image
import numpy as np
import torch

def process_image(image_path):
    image = Image.open(image_path)

    image = image.resize((256,256))
    image = image.crop((16,16,240,240))

    np_image = np.array(image)/255

    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])

    np_image = (np_image - means) / stds
    np_image = np_image.transpose((2,0,1))

    return torch.tensor(np_image).float()
