import json
import DirectoryManager as dm
import DataSets as dsets
import torch

class DataSourcesManager:
    def __init__(self, directoryManager):
        self.batch_size = 10 #55
        self.images_dataset_names = ['training_images','test_images', 'validation_images']
        print(directoryManager.get_working_directory_for_training_data())
        self.image_directories = {self.images_dataset_names[0]:directoryManager.get_working_directory_for_training_data(), 
                                  self.images_dataset_names[1]:directoryManager.get_working_directory_for_test_data(), 
                                  self.images_dataset_names[2]:directoryManager.get_working_directory_for_validation_data()
                                 }
        print(self.images_dataset_names[0])
        print(self.image_directories[self.images_dataset_names[0]])
    def get_images_dataset_names(self):
        return self.images_dataset_names
    
    def get_image_directories(self):
        return self.image_directories

def get_image_datasets (data_transforms, dataSourcesManager):
    datasets = dsets.DataSets(data_transforms, dataSourcesManager)
    image_datasets = datasets.get_data_sets()
    return image_datasets 

def get_dataloaders(transformsLoc, image_datasets, batch_size):
    dataloaders = {name: torch.utils.data.DataLoader(image_datasets[name], 
                   batch_size, shuffle=True) 
                   for name in transformsLoc.images_dataset_names}
    return dataloaders

def get_cat_to_name(json_file_path):
    if '.json' in json_file_path:
       json_file_path = json_file_path
    else:
       json_file_path = json_file_path + '/cat_to_name.json'
    
    with open(json_file_path, 'r') as f:
             cat_to_name = json.load(f)
    return cat_to_name

#directoryManager = dm.DirectoryManager()
#dataSourcesManager = DataSourcesManager(directoryManager)