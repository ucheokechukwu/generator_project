from keras import Model
from keras.layers import Input
from src.decorators import access_layer_properties

class BaseModel(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def build_graph(self, input_shape):
        input_ = Input(shape=input_shape)[1:]
        output_ = self.call(input_)
        return Model(inputs=input_, outputs=output_)
    
    @access_layer_properties
    def summary(self, model, **kwargs):
        return model.summary(**kwargs)

