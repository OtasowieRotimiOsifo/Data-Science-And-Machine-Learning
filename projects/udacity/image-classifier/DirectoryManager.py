import os
#data_dir = 'flowers'
#train_dir = data_dir + '/train'
#valid_dir = data_dir + '/valid'
#test_dir = data_dir + '/test'

batch_size = 10 #55
#data_dir = './flowers'
#train_dir = data_dir + '/train/'
#validation_dir = data_dir + '/valid/'
#test_dir = data_dir + '/test/'


# Define some utility classes
channels_mean_sequence = [0.485, 0.456, 0.406]
channels_standard_deviation_sequence = [0.229, 0.224, 0.225]
dimensions = [224, 224]
extra_dimensions = [256, 256]
class DirectoryManager:
    def __init__(self):
        self.root_directory = os.getcwd()
        self.working_directory = self.root_directory
        #For tests in google colab
        #self.data_dir = '/content/gdrive/My Drive/DataScience/p2_image_classifier/flowers'
        self.data_dir = self.root_directory + '/flowers'
        #self.data_dir = './flowers'
        self.save_state_path = self.root_directory + '/saved-state.pth'
    
    def set_data_directory(self, data_directory):
        c = data_directory[-1:]
        #print(c)
        if c != '/':
           self.data_dir = data_directory + '/'
        else:
           self.data_dir = data_directory
        
    def set_saved_state_path(self, save_dir):
        self.save_state_path = save_dir 
        
    def get_root_working_directory(self):
        return self.root_directory
    
    def change_to_data_directory(self, directory):
        os.chdir(directory)
        self.working_directory = directory
        
    def get_working_directory_for_training_data(self):
        c = self.data_dir[-1:]
        #print(c)
        if c != '/':
          self.training_data_dir = self.data_dir + '/train/'
        else:
          self.training_data_dir = self.data_dir + 'train/'
        return self.training_data_dir
        
    def get_working_directory_for_validation_data(self): 
        c = self.data_dir[-1:]
        #print(c)
        if c != '/':
           self.validation_data_dir = self.data_dir + '/valid/'
        else:
           self.validation_data_dir = self.data_dir + 'valid/'
        return self.validation_data_dir 
        
    def get_working_directory_for_test_data(self): 
        c = self.data_dir[-1:]
        #print(c)
        if c != '/':
           self.test_data_dir = self.data_dir + '/test/'
        else:
           self.test_data_dir = self.data_dir + 'test/'
        print('Working Directory')
        print(self.working_directory)
        print('test dir')
        print(self.test_data_dir)
        return self.test_data_dir
        
    def getRandomImageFromTestDirectory(self):
        self.get_working_directory_for_test_data()
        self.change_to_data_directory(self.test_data_dir)
        dir_name = np.random.choice(os.listdir(self.test_data_dir))
        #print('dir name')
        #print(dir_name)
        image_dir_path = self.test_data_dir + dir_name + '/'
        self.image_path = image_dir_path  + np.random.choice(os.listdir(image_dir_path))
        self.change_to_data_directory(self.root_directory)
        return self.image_path
        #print('path')
        #print(self.image_path)
    
    def setPredictImagePath(self, image_path):
        self.image_path = image_path
    
    def getImageClassFromPathName(self, imagePathName):
        return imagePathName.split('/')[-2]
    
    def get_saved_state_path(self):
        return self.save_state_path