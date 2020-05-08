import json
import matplotlib.pyplot as plt
from prettytable import PrettyTable

plt.rcParams['font.size'] = 14

class Presenter:
    def __init__(self, directoryManager):
        self.training_losses = []
        self.validation_losses = []
        self.training_accuracy = []
        self.validation_accuracy = []
        
        json_file_path = directoryManager.root_directory + '/cat_to_name.json'
        with open(json_file_path, 'r') as f:
             self.cat_to_name = json.load(f)
        
    def addlosses(self, losses, work_flow_phase, epoch):
        if work_flow_phase == 'training_images':
           tripple = (work_flow_phase, epoch, losses)
           self.training_losses.append(tripple)
        elif work_flow_phase == 'validation_images':
           tripple = (work_flow_phase, epoch, losses)
           self.validation_losses.append(tripple)
            
    def addaccuracy(self, accuracy, work_flow_phase, epoch):
        if work_flow_phase == 'training_images':
           tripple = (work_flow_phase, epoch, accuracy)
           self.training_accuracy.append(tripple)
        elif work_flow_phase == 'validation_images':
           tripple = (work_flow_phase, epoch, accuracy)
           self.validation_accuracy.append(tripple)
            
    def plotlosses(self):
        training_losses = []
        validation_losses = []
        epochs = []
        first = self.training_losses[0]
        training_phase = first[0]
        for tripple in self.training_losses:
            epochs.append(tripple[1])
            training_losses.append(tripple[2])
        
        first = self.validation_losses[0]
        validation_phase = first[0]
        for tripple in self.validation_losses:
            validation_losses.append(tripple[2])
        
        plt.plot(epochs, training_losses , 'r--', label=training_phase)
        plt.plot(epochs, validation_losses, 'g^', label=validation_phase)
        plt.show()
    
    def plotaccuracy(self):
        training_accuracies = []
        validation_accuracies = []
        epochs = []
        first = self.training_accuracy[0]
        training_phase = first[0]
        for tripple in self.training_accuracy:
            epochs.append(tripple[1])
            training_accuracies.append(tripple[2])
        
        first = self.validation_accuracy[0]
        validation_phase = first[0]
        for tripple in self.validation_accuracy:
            validation_accuracies.append(tripple[2])
        
        plt.plot(epochs, training_accuracies, label=training_phase)
        plt.plot(epochs, validation_accuracies, 'bs', label=validation_phase)
        plt.show()
    
    def print_table(self, top_probabilities, top_labels):
        print()
        idx = 0
        print('Table of results for top {} labels from the classifier'.format(len(top_labels)))
        prettyTable = PrettyTable()
        prettyTable.field_names = ["Predicted Flower Name", "Predicted Label", "Predicted Probability"]
        for label in top_labels:
            name = self.cat_to_name[label]
            prettyTable.add_row([name, label, top_probabilities[idx]])
            #print(label, empty, name, empty, top_probabilities[idx])
            idx = idx + 1
        print(prettyTable)
                                 
    def get_topk_names(self, top_labels):
        idx = 0
        names = []
        for label in top_labels:
            name = cat_to_name[label]
            names.append(name)
            idx = idx + 1
        return names
    
    def get_name_from_label(self, label):
        name = self.cat_to_name[label]
        #print(name)
        return name
            
    def display_prediction(self, imageProcessor, image_tensor, top_p, top_classes, real_class):
        names = self.get_topk_names(top_classes)
        
        #result = pd.DataFrame({'p': top_p}, index=top_classes) 
        result = pd.DataFrame({'p': top_p}, index=names)  
         
        plt.figure(figsize=(16, 5))
        ax = plt.subplot(1, 2, 1)
        ax, image = imageProcessor.imshow_tensor(image_tensor, ax=ax)

        # Set title to be the actual class
        name = self.get_name_from_label(real_class)
        #ax.set_title(real_class, size=20)
        ax.set_title(name, size=20)

        ax = plt.subplot(1, 2, 2)
        # Plot a bar plot of predictions
        result.sort_values('p')['p'].plot.barh(color='blue', edgecolor='k', ax=ax)
        plt.xlabel('Predicted Probability')
        plt.ylabel('Predicted Class')
        plt.tight_layout()
            
    def print_results(self, topk_probabilities,top_classes,imageClass):
        print()
        print('Top {} Probabilities'.format(len(top_classes)))
        print(topk_probabilities)
        print('Top {} probable iamges'.format(len(top_classes)))
        print(top_classes)
        name = self.get_name_from_label(top_classes[0])
        print('Selected Image for flower with name = ', name, 'and label = ', top_classes[0])
        #print(name)