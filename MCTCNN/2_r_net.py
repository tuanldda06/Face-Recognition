from tensorflow.keras import layers , Model

def build_rnet() :
    inputs = layers.Input(shape = (24,24,3))
    
    x =layers.Conv2D(28 , 3 , padding = 'valid')(inputs)
    x = layers.PReLU(shared_axis = [1,2])(x)
    x = layers.MaxPooling2D(pool_size = 3 , strides = 2 , padding = 'same')(x)
    
    
    x = layers.Conv2D(48 , 3 , padding ='valid')(x)
    x = layers.PReLU(shared_axis = [1,2])
    x = layers.MaxPooling2D(pool_size = 3 , strides = 2 , padding = 'same')(x)
    
    x = layers.Conv2D(64 , 2 , padding = 'valid')(x)
    x = layers.PReLU(shared_axis = [1,2])
    x = layers.Flatten(x)
    x = layers.Dense(128)(x)
    x = layers.PReLU(x)
    
    
    face = layers.Dense(2,activation = 'softmax')(x)
    bbox = layers.Dense(4,activation = 'linear')(x)
    
    return Model(inputs , {'face': face , 'bbox': bbox})
    
    