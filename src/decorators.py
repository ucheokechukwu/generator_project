from functools import wraps

def access_layer_properties(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.built:
            input_shape = self._build_input_shape[1:]
            model = self.build_graph(input_shape)
            
        else:
            raise ValueError("This model has not yet been built. Build the model first by calling `build()`.")
        return func(self, model, *args, **kwargs)
    
    
    return wrapper