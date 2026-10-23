from tensorflow.keras import layers , Model

def build_onet():
    inputs = layers.Input(shape = (48,48,3))
    
    x = layers.Conv2D(32 ,3 , padding = 'valid')(inputs)
    x = layers.PReLU(shared_axis = [1,2])(x)
    x = layers.MaxPooling2D(pool_size = 3, strides = 2 , padding = 'same')(x)
    
    
    x = layers.Conv2D(64 , 3 , padding = 'valid')(x)
    x = layers.PReLU(shared_axis = [1,2])(x)
    x = layers.MaxPooling2D(pool_size = 2 , strides = 2 , padding = 'valid')(x)
    
    
    x = layers.Conv2D(64 , 3 , padding = 'valid')(x)
    x = layers.PReLU(shared_axis = [1,2])(x)
    x = layers.MaxPooling2D(pool_size = 2 , strides = 2 , padding = 'valid')(x)
    
    
    x = layers.Conv2D(128 , 2 , padding = 'valid')(x)
    x = layers.PReLU(shared_axis = [1,2])
    x = layers.Flatten(x)
    x = layers.Dense(256)(x)
    x = layers.PReLU(x)
    
    face = layers.Dense(2 , activation = 'softmax')(x)
    bbox = layers.Dense(4 , activation = 'linear ')(x)
    landmarks = layers.Dense(10 , activation = 'linear')(x)
    
    
    return Model(inputs , {'face':face , 'bbox': bbox , 'landmarks': landmarks})