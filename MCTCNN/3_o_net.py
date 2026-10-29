from tensorflow.keras import layers , Model

def build_onet():
    inputs = layers.Input(shape = (48,48,3))
    
    # 48 x 48 x 3 thành 23 x 23 x 32
    x = layers.Conv2D(32 ,3 , padding = 'valid')(inputs)
    x = layers.PReLU(shared_axis = [1,2])(x)
    x = layers.MaxPooling2D(pool_size = 3, strides = 2 , padding = 'same')(x)
    
    
    # 23 x 23 x 32 trở thành 10 x 10 x 64
    x = layers.Conv2D(64 , 3 , padding = 'valid')(x)
    x = layers.PReLU(shared_axis = [1,2])(x)
    x = layers.MaxPooling2D(pool_size = 2 , strides = 2 , padding = 'valid')(x)
    
    # 10 x 10 x 64 trở thành 4 x 4 x 64
    x = layers.Conv2D(64 , 3 , padding = 'valid')(x)
    x = layers.PReLU(shared_axis = [1,2])(x)
    x = layers.MaxPooling2D(pool_size = 2 , strides = 2 , padding = 'valid')(x)
    
    # 4 x 4 x 64 trở thành 3 x 3 x 128
    x = layers.Conv2D(128 , 2 , padding = 'valid')(x)
    x = layers.PReLU(shared_axis = [1,2])
    
    #Trải phẳng
    x = layers.Flatten(x)
    x = layers.Dense(256)(x)
    x = layers.PReLU(x)
    
    face = layers.Dense(2 , activation = 'softmax')(x)
    bbox = layers.Dense(4 , activation = 'linear ')(x)
    landmarks = layers.Dense(10 , activation = 'linear')(x)
    
    
    return Model(inputs , {'face':face , 'bbox': bbox , 'landmarks': landmarks})