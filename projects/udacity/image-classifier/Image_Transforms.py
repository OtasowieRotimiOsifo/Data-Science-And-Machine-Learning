from torchvision import transforms

class Transforms:
    def __init__(self, dataSourcesManager):
        self.channels_mean_sequence = [0.485, 0.456, 0.406]
        self.channels_standard_deviation_sequence = [0.229, 0.224, 0.225]
        self.dimensions = [224, 224]
        self.extra_dimensions = [256, 256]
        self.images_dataset_names = dataSourcesManager.get_images_dataset_names()
        #print(self.images_dataset_names[0])
        self.data_transforms = {
                                self.images_dataset_names[0]: transforms.Compose([
                                                     transforms.RandomRotation(35),
                                                     transforms.RandomResizedCrop(224),
                                                     transforms.RandomHorizontalFlip(),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize(self.channels_mean_sequence, 
                                                     self.channels_standard_deviation_sequence)
                                ]),
    
                                self.images_dataset_names[1]: transforms.Compose([
                                                     transforms.Resize(self.extra_dimensions), 
                                                     transforms.CenterCrop(self.dimensions), 
                                                     transforms.ToTensor(),
                                                     transforms.Normalize(self.channels_mean_sequence, 
                                                     self.channels_standard_deviation_sequence)
                                ]),
    
                                self.images_dataset_names[2]: transforms.Compose([
                                                     transforms.Resize(self.extra_dimensions),
                                                     transforms.CenterCrop(self.dimensions), 
                                                     transforms.ToTensor(),
                                                     transforms.Normalize(self.channels_mean_sequence, 
                                                     self.channels_standard_deviation_sequence)
                                ])
                               }
        
    def get_transforms(self):
        return self.data_transforms
    
    def get_transforms_for_training_data(self):
        key = self.images_dataset_names[0]
        return self.data_transforms[key]
    
    def get_transforms_for_validation_data(self):
        key = self.images_dataset_names[1]
        return self.data_transforms[key]
    
    
    def get_transforms_for_test_data(self):
        key = self.images_dataset_names[2]
        return self.data_transforms[key]
    