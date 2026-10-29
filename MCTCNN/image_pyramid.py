import numpy as np
from PIL import Image

from image_processing import check_rgb


#check ảnh
def build_image_pyramid(image , min_face_size = 20.0 , factor = 0.79):
    
    image = check_rgb(image)
    
    height , width = image.shape[:2]
    scale = 12.0 / min_face_size
    
    source = Image.fromarray(image) 
    
    while min(height , width) * scale >= 12.0 :
        size = (int(np.ceil(width * scale)) , int(np.ceil(height * scale)))
        scaled = np.asarray(source.resize(size , Image.Resampling.BILINEAR) , dtype = np.unit8)
        yield scale , scaled
        scale *= factor
    