from torchvision import datasets
import torch

class DataSets:
    def __init__(self, data_transforms, dataSourcesManager):
        self.image_directories = dataSourcesManager.get_image_directories()
        self.images_dataset_names = dataSourcesManager.get_images_dataset_names()
        
        self.datasets = {name: datasets.ImageFolder(self.image_directories[name],   
                         transform=data_transforms[name]) 
                         for name in self.images_dataset_names}
    
    def get_data_sets(self):
        return self.datasets
    
    def get_training_dataset(self):
        key = self.images_dataset_names[0]
        return self.datasets[key]
    
    def get_validation_dataset(self):
        key = self.images_dataset_names[1]
        return self.datasets[key]
    
    def get_test_dataset(self):
        key = self.images_dataset_names[2]
        return self.datasets[key]