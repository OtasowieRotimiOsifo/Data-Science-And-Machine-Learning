import matplotlib.pyplot as plt
from PIL import Image
import torch
import json
import os
import time
import copy
import pandas as pd
import numpy as np
import argparse
import torch.nn.functional as F
from torch import nn as nn
from torch import optim as optim
from torch import cuda as cuda
from torch.optim import lr_scheduler
from torch.autograd import Variable
#from torchsummary import summary
from torchvision import datasets, models, transforms
from timeit import default_timer as timer

#import of non-standard modules
import DirectoryManager as dm
import DataSourcesManager as dsm
import GpuEnvironment as gpue
import Presenter as presenter
import Image_Transforms as ts
import NeuralNetworkModelBuilder as nb
import ModelTrainer as mt
import ImageClassifier as ic
import Check_point_manager as cpm

def main():
    parser = argparse.ArgumentParser(prog='train')
    parser.add_argument('data_directory', type=str, help='The root directory where images are stored')
    parser.add_argument('--save_dir', type=str, help='The directory for storing the trained model')
    parser.add_argument('--arch', type=str, help='The transfer learning model to be used')
    parser.add_argument('--learning_rate', type=float, help='Learning rate for the model')
    parser.add_argument('--hidden_units', type=int, help='Number of hidden units in the fully connected layer')
    parser.add_argument('--epochs', type=int, help='The number of epochs paases that will be used for training the model')
    parser.add_argument('--gpu', type=str, help='Use GPU for predictions')
    args = parser.parse_args()
    
    data_directory = args.data_directory
    directoryManager = dm.DirectoryManager()
    directoryManager.set_data_directory(data_directory)
    
    dataSourcesManager = dsm.DataSourcesManager(directoryManager)
    gpuEnvironment = gpue.GpuEnvironment()
    dataPresenter = presenter.Presenter(directoryManager)
    transformsLoc = ts.Transforms(dataSourcesManager)
    data_transforms = transformsLoc.get_transforms()
    image_datasets = dsm.get_image_datasets (data_transforms, dataSourcesManager)
    dataloaders = dsm.get_dataloaders(transformsLoc, image_datasets, dataSourcesManager.batch_size)
    
    json_file_path = directoryManager.get_root_working_directory()
    cat_to_name = dsm.get_cat_to_name(json_file_path)
    dataset_sizes = {name: len(image_datasets[name]) 
                     for name in transformsLoc.images_dataset_names}
    class_names = image_datasets['training_images'].classes
    
    save_dir = args.save_dir
    if save_dir != None:
       directoryManager.set_saved_state_path(save_dir)
    #print(directoryManager.save_state_path) 
    
    
    saved_state_path = directoryManager.get_saved_state_path()
    print(saved_state_path)
    
    loss_function = nb.get_loss_function()
    arch = args.arch
    if arch == None:
       arch = 'VGG19'
   
    hidden_layers = args.hidden_units
    cnnModelBuilder = nb.NeuralNetworkModelBuilder(arch, loss_function, from_saved=False)
    if hidden_layers != None:
       cnnModelBuilder.set_number_of_hidden_layers(hidden_layers)
   
    cnnModelBuilder.buildAndAddFullyConnectedLayer(cnnModelBuilder.modelName,
                                                   image_datasets['training_images'],
                                                   cat_to_name
                                                  )
    
    gpu = args.gpu
    if gpu != None:
        gpuEnvironment.set_use_gpu(gpu)
        
    modelTrainer = mt.ModelTrainer(cnnModelBuilder.model, loss_function, dataloaders, dataSourcesManager.batch_size)
    lr = args.learning_rate
    if lr != None:
        modelTrainer.set_learning_rate(lr)
    
    epochs = args.epochs
    if epochs != None:
       modelTrainer.set_epochs(epochs)
     
    print(modelTrainer.device)
    modelTrainer.trainModel(dataPresenter)

    #Validate with test data 
    modelTrainer.validate_model_with_test_data()
    #testdataloader = dataloaders['test_images']
    
    #bestModel = modelTrainer.model
    #bestModel.load_state_dict(modelTrainer.best_model_wts)
    #classifier = ic.ImageClassifier(bestModel, modelTrainer.device)
    #classifier.classify(testdataloader, cuda=True)

    #Save the Trained Model for future use
    
    check_point_manager = cpm.Check_point_manager()
    check_point_manager.set_values(cnnModelBuilder.modelName,
                                   modelTrainer.best_model_wts,
                                   cnnModelBuilder.model.classifier,
                                   modelTrainer.optimizer,
                                   cnnModelBuilder.model
                                  )
    check_point_manager.save_state(path=saved_state_path)
    
if __name__ == '__main__':
    main()















        






