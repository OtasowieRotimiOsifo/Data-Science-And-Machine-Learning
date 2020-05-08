import torch

class Check_point_manager:
    def __init__(self):
        self.saved_state = dict()               
        
    
    def set_values(self, modelName, best_model_wts, linear_classifier,optimizer, model):
        self.saved_state['arch_name'] = modelName
        self.saved_state['best_model_wts'] = best_model_wts
        self.saved_state['linear_classifier'] = linear_classifier
        self.saved_state['optimizer_state'] = optimizer.state_dict()
        self.saved_state['class_to_idx'] = model.class_to_idx
        self.saved_state['idx_to_class'] = model.idx_to_class
        self.saved_state['classes'] = model.classes
        self.saved_state['cat_to_name'] = model.cat_to_name
                           
    def save_state(self, path):
        torch.save(self.saved_state, path)
    
    def load_state(self, path):
        self.saved_state = torch.load(path)
