from tensorflow.keras import layers , Model

def build_rnet() :
    inputs = layers.Input(shape = (24,24,3))
    
    
    # từ 24 x 24 x 3 trở thành 11 x 11 x 28
    x =layers.Conv2D(28 , 3 , padding = 'valid')(inputs)
    x = layers.PReLU(shared_axis = [1,2])(x)
    x = layers.MaxPooling2D(pool_size = 3 , strides = 2 , padding = 'same')(x)
    
    
    # từ 11 x 11 x 28 thành 4 x 4 x 48
    x = layers.Conv2D(48 , 3 , padding ='valid')(x)
    x = layers.PReLU(shared_axis = [1,2])
    x = layers.MaxPooling2D(pool_size = 3 , strides = 2 , padding = 'same')(x)
    
    # từ 4 x 4 x 48 thành 3 x 3 x 64
    x = layers.Conv2D(64 , 2 , padding = 'valid')(x)
    x = layers.PReLU(shared_axis = [1,2])
    
    #Trải phẳng
    x = layers.Flatten(x)
    x = layers.Dense(128)(x)
    x = layers.PReLU(x)
    
    #
    face = layers.Dense(2,activation = 'softmax')(x)
    bbox = layers.Dense(4,activation = 'linear')(x)
    
    return Model(inputs , {'face': face , 'bbox': bbox})
    
    