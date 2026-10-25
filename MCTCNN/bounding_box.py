import numpy as np


#Chạy ngay sau P-Net , face_map là kết quả đầu ra của lớp tích chập 
def generate_bounding_box(face_map ,bbox_map , scale , threshold = 0.6):
    probabilities = np.asarray(face_map , dtype = np.float32)
    offsets = np.asarray(bbox_map , dtype = np.float32)
    
    rows , columns = np.where(probabilities > 0.6) # Xét cả cặp row , columns xem cặp nào t/m trả về index cặp đó trong ảnh đã qua P-Net
    boxes = np.stack([2 * columns , 2* rows , 2 * columns + 12 , 2 * rows + 12] , axis = 1)
    return (boxes.astype(np.float32)) / scale , probabilities[rows , columns] , offsets[rows , columns]

def valid_box_indices(boxes):
    boxes = np.asarray(boxes)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes phải có shape (N,4).")
    mask = np.all(np.isfinite(boxes), axis=1) & np.all(boxes[:, 2:] > boxes[:, :2], axis=1)
    return np.flatnonzero(mask)



