
from PIL import Image

class ImageClassificationDriver:
    def __init__(self, model, directoryManager):
        self.imagePath = directoryManager.image_path
        self.image = Image.open(self.imagePath)
        self.model = model
        self.imageClass = None #directoryManager.getImageClassFromPathName(self.imagePath)
        #print('iamge_path')
        #print(image_path)
    
    def set_image_class(self, cat_to_name, image_name):
        keys = cat_to_name.keys()
        for key in keys:
            name = cat_to_name[key]
            if name in image_name:
               self.imageClass = key
               break
    def driveClassification(self, imageClassifier, imageProcessor, gpuEnvironment, topk_selected_predictions):
        imageClassifier.predict_image(
                                      imageProcessor,
                                      self.image, 
                                      self.model, 
                                      gpuEnvironment.usegpu, 
                                      gpuEnvironment.device, 
                                      topk_selected_predictions
                                     )
         
    def display_results(self, presenter, imageProcessor, imageClassifier):
         presenter.display_prediction(
                                    imageProcessor,
                                    imageClassifier.image_tensor, 
                                    imageClassifier.topk_probabilities, 
                                    imageClassifier.top_classes, 
                                    self.imageClass
                                   )
    def print_results(self, presenter, imageClassifier):
        presenter.print_results(imageClassifier.topk_probabilities,
                                imageClassifier.top_classes,
                                self.imageClass
                               )

    def print_table(self, presenter, imageClassifier):
        presenter.print_table(imageClassifier.topk_probabilities, imageClassifier.top_classes)