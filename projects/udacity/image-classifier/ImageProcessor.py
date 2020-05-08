import torch
import numpy as np
#From Udacity
#def imshow(image, ax=None, title=None):
#    """Imshow for Tensor."""
#    if ax is None:
#        fig, ax = plt.subplots()
#    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
#    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
#    mean = np.array([0.485, 0.456, 0.406])
#    std = np.array([0.229, 0.224, 0.225])
#    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
#    image = np.clip(image, 0, 1)
    
#    ax.imshow(image)
    
#    return ax

class ImageProcessor:
    def __init__(self):
        print('Initializing the image processor')

    def imshow_tensor(self, image, ax=None, title=None):
        if ax is None:
           fig, ax = plt.subplots()
        
        image = image.numpy().transpose((1, 2, 0))

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

        ax.imshow(image)
        plt.axis('off')

        return ax, image

    def imshow(self, image):
        plt.figure(figsize=(6, 6))
        plt.imshow(image)
        plt.axis('off')
        plt.show()
    
    def preProcessor(self,image):
         mean = np.array([0.485, 0.456, 0.406])
         std = np.array([0.229, 0.224, 0.225])
         initial_image_width = 256
         initial_image_height  = 256
         new_image = image.copy()
         new_image.thumbnail((initial_image_width ,initial_image_height)) #done in place
    
         expected_width = 224
         expected_height = 224

         left = (initial_image_width  - expected_width) / 2
         top = (initial_image_height - expected_height) / 2
         right = (initial_image_width + expected_width) / 2
         bottom = (initial_image_height + expected_height ) / 2
         new_image = new_image.crop((left, top, right, bottom))

    
         new_image = np.array(new_image).transpose((2, 0, 1)) / 256

         means = np.array(mean).reshape((3, 1, 1))
         stds = np.array(std).reshape((3, 1, 1))

         new_image = new_image - means
         new_image = new_image / stds

         image_as_tensor = torch.Tensor(new_image)

         return image_as_tensor
    