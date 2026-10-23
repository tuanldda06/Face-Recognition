from tensorflow.keras import Model ,layers

def build_pnet() :
    inputs = layers.Input(shape = (None , None , 3))
    
    # Với patch 12x12: 12x12x3 -> 10x10x10.
    x = layers.Conv2D(10 , 3 ,  padding = 'valid')(inputs)#padding = 'valid' có nghĩa là không sử dụng lớp đệm (no padding)
    x = layers.PReLU(shared_axis =[1,2])(x)
    
    # 10x10x10 -> 5x5x10.
    x = layers.MaxPooling2D(
        pool_size = 2, strides = 2 , padding = 'same'
    )(x)
    
    # 5x5x10 -> 3x3x16.
    x = layers.Conv2D(16,3 , padding = 'valid')(x)
    x = layers.PReLU(shared_axis = [1,2])
    
    
    # 3x3x16 -> 1x1x32.
    x = layers.Conv2D(32,3 , padding = 'valid')
    x = layers.PReLU(shared_axis = [1,2])
    
     # Hai nhánh dùng chung feature map x, mỗi nhánh có trọng số riêng.
    face = layers.Conv2D(
         2,1 , activation ='softmax'
     )
    
    bbox = layers.Conv2D(
        4,1,activation = 'linear'
    )
    
    return Model(inputs , {'face': face , 'bbox': bbox})

    
    
    
    
