import torch

class ImageClassifier:
    def __init__(self, model, device):
        self. model = model
        self.device = device
        self.image_tensor = None
        self.top_p = None
        self.top_classes = None
        
        self.model.eval()
        #self.loss_function = loss_function
    
     #Based on an article and knowledge from Josh Bernhard @ Medium
    def classify(self,testdataloader, cuda=False):
       
        self.model.to(self.device)    
         
        with torch.set_grad_enabled(False):
             for idx, (inputs, labels) in enumerate(testdataloader):   
                if cuda:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model.forward(inputs)

                _, predicted = outputs.max(dim=1)
 
                if idx == 0:
                    print(predicted)
                    print(torch.exp(_))
                equals = predicted == labels.data

                if idx == 0:
                    print(equals)

                print(equals.float().mean())
    
    def predict_image(self, imageProcessor, image, model, usegpu, device, topk_selected_predictions=5):
        image_as_tensor = imageProcessor.preProcessor(image)
        model.to(device)
        
        #imageProcessor.imshow_tensor(image)
        #imageProcessor.imshow(image)
        print(image_as_tensor.shape)
        if usegpu:
           #img_as_tensor = image_as_tensor.view(1, 3, 224, 224).to(device)
           image_as_tensor = image_as_tensor.unsqueeze(0).to(device)
        else:
           image_as_tensor = image_as_tensor.unsqueeze(0)

        with torch.no_grad():
             model.eval()
        
             out = model.forward(image_as_tensor)
             predictions = torch.exp(out)

             topk_selected_predictions, top_labels = predictions.topk(topk_selected_predictions, dim=1)

        top_classes = [
            model.idx_to_class[class_] for class_ in top_labels.cpu().numpy()[0]
        ]
        
        topk_probabilities = topk_selected_predictions.cpu().numpy()[0]

        self.image_tensor = image_as_tensor.cpu().squeeze()
        self.topk_probabilities = topk_probabilities
        self.top_classes = top_classes