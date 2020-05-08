from torch import nn as nn
from torchvision import models
import DataSourcesManager as dsm

class NeuralNetworkModelBuilder:
    def __init__(self, modelName, loss_function, from_saved):
        self.loss_function = loss_function
        self.hidden_layers = 4096
        if from_saved == True:
            self.model_from_saved_state = self.initialize(modelName)
            for param in self.model_from_saved_state.parameters():
                param.requires_grad_(False)
        else:
            self.model= self.initialize(modelName)
            for param in self.model.parameters():
                param.requires_grad_(False) 
    
    def set_number_of_hidden_layers(self, hidden_layers):
        self.hidden_layers = hidden_layers
        
    def getPretrainedCNNModel(self):
        return self.model
    
    def initialize(self, modelName):
        self.n_outputs = 102
        self.modelName = modelName
        
        if modelName != None and modelName == "AlexNet":
            return models.alexnet(pretrained=True)
        elif modelName != None and modelName == "VGG16":
            return models.vgg16(pretrained=True)
        elif modelName != None and modelName == "VGG13":
            return models.vgg13(pretrained=True)
        elif modelName != None and modelName == "VGG19":
            return models.vgg19(pretrained=True)
        else:
            return models.vgg19(pretrained=True)
    
    def modelSummary(self, model,device):
        model.to(device)
        summary(model, 
                input_size=(3, 224, 224), 
                batch_size=batch_size,
                device="cuda" 
               )
        
    def buildAndAddFullyConnectedLayer(self, modelName, training_images_dataset, cat_to_name):
        if modelName != None and modelName == "VGG13":
           number_of_inputs = self.model.classifier[0].in_features
           fclayer1 = nn.Linear(number_of_inputs, self.hidden_layers)
           relu = nn.ReLU()
           fclayer2 = nn.Linear(self.hidden_layers, self.n_outputs)
           output = nn.LogSoftmax(dim=1)
           #dropout = nn.Dropout(0.3)
           classifier = nn.Sequential(
                                      fclayer1 ,
                                      relu,
                                      #dropout,
                                      fclayer2,
                                      output
                                     )
           #self.model.classifier[6] = classifier
           self.model.class_to_idx = training_images_dataset.class_to_idx
           self.model.classes = training_images_dataset.classes
           self.model.idx_to_class = {
                                      idx: class_
                                      for class_, idx in self.model.class_to_idx.items()
                                     }
           self.model.cat_to_name = cat_to_name
           self.model.classifier = classifier
        elif modelName != None and modelName == "VGG16": 
             number_of_inputs = self.model.classifier[0].in_features
             fclayer1 = nn.Linear(number_of_inputs, self.hidden_layers)
             relu = nn.ReLU()
             fclayer2 = nn.Linear(self.hidden_layers, self.n_outputs)
             output = nn.LogSoftmax(dim=1)
             #dropout = nn.Dropout(0.3)
             classifier = nn.Sequential(
                                       fclayer1 ,
                                       relu,
                                       #dropout,
                                       fclayer2,
                                       output
                                       )
             #self.model.classifier[6] = classifier
             self.model.class_to_idx = training_images_dataset.class_to_idx
             self.model.classes = training_images_dataset.classes
             self.model.idx_to_class = {
                                       idx: class_
                                       for class_, idx in self.model.class_to_idx.items()
                                      }
             self.model.cat_to_name = cat_to_name
             self.model.classifier = classifier
        elif modelName != None and modelName == "VGG19":
            number_of_inputs = self.model.classifier[0].in_features
            fclayer1 = nn.Linear(number_of_inputs, self.hidden_layers)
            relu = nn.ReLU()
            fclayer2 = nn.Linear(self.hidden_layers, self.n_outputs)
            output = nn.LogSoftmax(dim=1)
            dropout = nn.Dropout(0.3)
            classifier = nn.Sequential(
                                       fclayer1 ,
                                       relu,
                                       #dropout,
                                       fclayer2,
                                       output
                                       )
            #self.model.classifier[6] = classifier
            
            self.model.class_to_idx = training_images_dataset.class_to_idx
            self.model.classes = training_images_dataset.classes
            self.model.idx_to_class = {
                                       idx: class_
                                       for class_, idx in self.model.class_to_idx.items()
                                      }
            self.model.classifier = classifier
            self.model.cat_to_name = cat_to_name
        elif modelName != None and modelName == "AlexNet":
            number_of_inputs = self.model.classifier[0].in_features
            fclayer1 = nn.Linear(number_of_inputs, self.hidden_layers)
            relu = nn.ReLU()
            fclayer2 = nn.Linear(self.hidden_layers, self.n_outputs)
            output = nn.LogSoftmax(dim=1)
            dropout = nn.Dropout(0.3)
            classifier = nn.Sequential(
                                       fclayer1 ,
                                       relu,
                                       #dropout,
                                       fclayer2,
                                       output
                                       )
            self.model.classifier = classifier
  
    def buildModelFromSavedState(self, check_point_manager):
        saved_state = check_point_manager.saved_state
        arch_name = saved_state['arch_name']
        self.model_from_saved_state = self.initialize(arch_name)
        
        for param in self.model_from_saved_state.parameters():
            param.requires_grad_(False)
            
        self.model_from_saved_state.classifier = saved_state['linear_classifier']
        self.model_from_saved_state.load_state_dict(saved_state['best_model_wts'])
        self.model_from_saved_state.classes = saved_state['classes']
        self.model_from_saved_state.class_to_idx = saved_state['class_to_idx']
        self.model_from_saved_state.idx_to_class = saved_state['idx_to_class']
        self.model_from_saved_state.cat_to_name = saved_state['cat_to_name']
        #self.loss_function = saved_state['loss_function']

def get_loss_function():
    loss_function = nn.NLLLoss()
    return loss_function

#cnnModelBuilder = NeuralNetworkModelBuilder('VGG19', get_loss_function(), from_saved=False)
#print(cnnModelBuilder.model)

#image_datasets = dsm.image_datasets['training_images'],
#cnnModelBuilder.buildAndAddFullyConnectedLayer(cnnModelBuilder.modelName,
#                                               image_datasets['training_images'],
#                                               cat_to_name
#                                              )