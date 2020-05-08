import torch
import time

from torch import optim as optim
from torch.optim import lr_scheduler
import copy

class ModelTrainer:
    def __init__(self, model, loss_function, dataloaders, batch_size):
        self.batch_size = batch_size
        self.dataloaders = dataloaders
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(self.device)
        self.model = model
        self.learning_rate = 0.0001
        
        self.loss_function = loss_function
        self.optimizer = optim.Adam(self.model.classifier.parameters(), self.learning_rate)
        self.epochs = 15
        self.phases = ['training_images','validation_images']
        self.epoch_loss = 0.0
        self.epoch_corrects = 0
        self.best_acc = 0.0
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=4, gamma=0.1)
        self.current_loss = 0.0
        self.current_currects = 0
        self.best_model_wts = copy.deepcopy(self.model.state_dict())
        #self.analyzer = Analyzer()
     
    def set_use_gpu(self, gpuEnvironment):
        self.device = gpuEnvironment.device
        self.model = self.model.to(self.device)
        
    #Based on knowledge from Josh Bernhard and others + Pytorch documentation
    def trainModel(self, presenter):
        since = time.time()
        
        for epoch in range(self.epochs):
            self.epoch_loss = 0.0
            self.epoch_corrects = 0
            print('Epoch {}/{}'.format(epoch + 1, self.epochs))
            print('-' * 10)

            # Each epoch has a training and validation phase
            self.train_in_one_epoch(epoch, presenter)              
                
            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
              time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(self.best_acc))

        # load best model weights
        self.model.load_state_dict(self.best_model_wts)
        
        #presenter.plotlosses()
        #presenter.plotaccuracy()
        return
    
     #Based on knowledge from Josh Bernhard and others + Pytorch documentation
    def train_in_one_epoch(self, epoch, presenter):
      
        for phase in self.phases:
            self.current_loss = 0.0
            self.current_currects = 0
            
            if phase == 'training_images':
               #dataLoader = self.train_dataloader
               self.scheduler.step()
               self.model.train()  # Set model to training mode
            else:
               self.model.eval()   # Set model to evaluate mode
              
            for datapair in self.dataloaders[phase]:
                data, targets = datapair
                data = data.to(self.device)
                targets = targets.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'training_images'):
                     outputs = self.model(data)
                     _, predictions = torch.max(outputs.data, dim=1)
                     self.loss = self.loss_function(outputs, targets)

                     # backward + optimize only if in training phase
                     if phase == 'training_images':
                        self.loss.backward()
                        self.optimizer.step()

                self.current_loss += self.loss.item() * data.size(0)
                self.current_currects += torch.sum(predictions == targets.data)
                   
            self.epoch_loss = self.current_loss / len(self.dataloaders[phase].dataset)
            self.epoch_corrects = self.current_currects.double() / len(self.dataloaders[phase].dataset) #dataset_sizes[phase]
            
            presenter.addlosses(self.epoch_loss, phase, epoch + 1)
            presenter.addaccuracy(self.epoch_corrects, phase, epoch + 1)
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                      phase, self.epoch_loss, self.epoch_corrects))

            # deep copy the model
            if phase == 'validation_images' and self.epoch_corrects > self.best_acc:
               self.best_acc = self.epoch_corrects
               self.best_model_wts = copy.deepcopy(self.model.state_dict())
               #self.model.load_state_dict(self.best_model_wts)
            stmt2 = 'exiting the training loop for epoch: '
            print(stmt2 + str(epoch+1))
        return
    
     #Based on knowledge from Josh Bernhard @ Medium and others + Pytorch documentation
    def validate_model_with_test_data(self):
        print('Validation of the trained model is starting')
        phase = 'test_images'
        self.model.eval()
        current_loss = 0.0
        current_corrects = 0
        self.model.to(self.device)
        dataloader = self.dataloaders[phase]
        for data, targets in dataloader:
            data = data.to(self.device)
            targets = targets.to(self.device)
            with torch.set_grad_enabled(False):
                outputs = self.model.forward(data)
                _, predictions = outputs.max(dim=1)
                loss = self.loss_function(outputs, targets)
            current_loss += loss.item() * data.size(0)
            current_corrects += torch.sum(predictions == targets.data)
        total_loss = current_loss / len(self.dataloaders[phase].dataset) #dataset_sizes[phase]
        total_corrects = current_corrects.double() /   len(self.dataloaders[phase].dataset) #len(self.dataloaders[phase].dataset)
        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
              phase, total_loss, total_corrects))
        return
    
    def set_epochs(self, epochs):
        self.epochs = epochs
        
    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate