from src.starter import *
from src.model_classes import *

# m.save_weights("/content/model_weights/from_colab_X0_WB6.weights.h5", overwrite=True)
m = AttentionModel()
m.build(input_shape =( None, 1300, 1))
m.load_weights("/content/model_weights/from_colab_X0_WB6.weights.h5", skip_mismatch=False)
m.summary()

data = sample_data(target='WB 6', 
                    horizon_x0=horizon, 
                    horizon_x4=0, 
                    horizon_x5=0, 
                    scaling=True)
X0_train = data[0]
X0_test = data[1]
y_train, y_test = data[-2], data[-1]
set_random_seed(0)
m.compile(loss='sparse_categorical_crossentropy', metrics='accuracy', optimizer='adamax')
m.evaluate(X0_test, y_test)




