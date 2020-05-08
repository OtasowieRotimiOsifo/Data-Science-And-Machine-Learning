import matplotlib.pyplot as plt
from PIL import Image
import argparse
import torch
import json
import os
import time
import copy
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch import nn as nn
from torch import optim as optim
from torch import cuda as cuda
from torch.optim import lr_scheduler
from torch.autograd import Variable
#from torchsummary import summary
from torchvision import datasets, models, transforms
from timeit import default_timer as timer

#import matplotlib.pyplot as plt
#%matplotlib inline
#plt.rcParams['font.size'] = 14

#import for non standard modules
import Check_point_manager as cpm
import NeuralNetworkModelBuilder as nb
import ImageClassifier as ic
import DirectoryManager as dm
import DataSourcesManager as dsm
import GpuEnvironment as gpue
import Presenter as presenter
import Image_Transforms as ts
import GpuEnvironment as gpue
import ImageProcessor as imp
import ImageClassificationDriver as imcd

#From Udacity
#def predict(image_path, model, topk=5):
#    ''' Predict the class (or classes) of an image using a trained deep learning model.
#    '''


def main():
    parser = argparse.ArgumentParser(prog='predict')
    parser.add_argument('path_to_image', type=str, help='The path to where the image to be predicted is stored')
    parser.add_argument('--checkpoint', type=str, help='The path to where the the check point is stored')
    parser.add_argument('--top_k', type=int, help='The top k probabilities computed during prediction')
    parser.add_argument('--category_names', type=str, help='The input file for mapping flower categories to flower names')
    parser.add_argument('--gpu', type=str, help='Use GPU for predictions')
    args = parser.parse_args()
        
    print('Loaded model från saved state')
    check_point_manager = cpm.Check_point_manager()
    checkpoint = args.checkpoint    
    check_point_manager.load_state(path=checkpoint)
    
    directoryManager = dm.DirectoryManager()
    
    dataSourcesManager = dsm.DataSourcesManager(directoryManager)
    dataPresenter = presenter.Presenter(directoryManager)
    imageProcessor = imp.ImageProcessor()
    
    loss_function = loss_function = nb.get_loss_function()
    cnnModelBuilder = nb.NeuralNetworkModelBuilder('VGG19', loss_function, from_saved=True)
    cnnModelBuilder.buildModelFromSavedState(check_point_manager)
    print(cnnModelBuilder.model_from_saved_state)
    
    gpuEnvironment = gpue.GpuEnvironment()
    gpu = args.gpu
    if gpu != None:
        gpuEnvironment.set_use_gpu(gpu)
    
    cat_to_name = args.category_names
    if args.category_names != None:
        cat_to_name = dsm.get_cat_to_name(args.category_names)
    
    path_to_image = args.path_to_image
    if path_to_image != None:
       path_to_image = directoryManager.setPredictImagePath(path_to_image)
        
    topk_selected_predictions = args.top_k 
    if topk_selected_predictions == None:
        topk_selected_predictions = 5
        
    classifier = ic.ImageClassifier(cnnModelBuilder.model_from_saved_state, gpuEnvironment.device)
    
    imageClassificationDriver = imcd.ImageClassificationDriver(
                                                          cnnModelBuilder.model_from_saved_state,
                                                          directoryManager 
                                                         )
    #imageClassificationDriver.set_image_class(cat_to_name, path_to_image)
    imageClassificationDriver.driveClassification(classifier, 
                                                  imageProcessor, 
                                                  gpuEnvironment, 
                                                  topk_selected_predictions
                                                 )
    
    #imageClassificationDriver.driveClassification()
    imageClassificationDriver.print_results(dataPresenter, classifier)

    # Display an image along with the top 5 classes
    #imageClassificationDriver.display_results()
    imageClassificationDriver.print_table(dataPresenter, classifier)


if __name__ == '__main__':
    main()



