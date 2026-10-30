import numpy as np


#Chạy ngay sau P-Net , face_map là kết quả đầu ra của lớp tích chập 
def generate_bounding_box(face_map ,bbox_map , scale , threshold = 0.6):
    rows , columns = np.where(face_map >= threshold)
    
    x1 = 2 * columns
    y1 = 2 * rows
    x2 = x1 + 12
    y2 = y1 + 12
    
    boxes = np.stack([x1 , y1 , x2 , y2] , axis = 1)
    boxes = boxes.astype(np.float32) / scale

    scores = face_map[rows , columns]
    offsets = bbox_map[rows , columns] # Bốn giá trị hồi quy do nhánh BBox Regression của P-Net/R-Net/O-Net dự đoán.
    return boxes , scores , offsets

#Tinh IoU và Non-Maximum Suppression
#Tinh IoU và Non-Maximum Suppression
def iou(box , boxes): # boxes :[x1, y1 , x2 , y2]
    x1 = np.maximum(box[0] , boxes[:,0])
    y1 = np.maximum(box[1] , boxes[:,1])
    x2 = np.minimum(box[2] , boxes[:,2])
    y2 = np.minimum(box[3] , boxes[:,3])
    
    w = np.maximum(0 , x2 - x1)
    h = np.maximum(0 , y2 - y1)
    
    intersection = w * h
    area1 = (box[2] - box[0])*(box[3]-box[1])
    area2 = (boxes[:,2] -  boxes[:,0]) * (boxes[:,3] - boxes[:,1])
    
    union = area1 + area2 - intersection
    
    return intersection / union


def nms(boxes, scores, threshold=0.5, mode="union"):
    """Remove overlapping boxes, keeping the highest scores."""
    order = np.argsort(scores)[::-1]
    areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)

        remaining = order[1:]
        if len(remaining) == 0:
            break

        # Tính diện tích giao nhau
        top_left = np.maximum(boxes[i, :2], boxes[remaining, :2])
        bottom_right = np.minimum(boxes[i, 2:], boxes[remaining, 2:])
        wh = np.maximum(0, bottom_right - top_left)
        intersection = wh[:, 0] * wh[:, 1]

        # Tính overlap
        if mode == "min":
            denominator = np.minimum(areas[i], areas[remaining])
        else:
            denominator = areas[i] + areas[remaining] - intersection

        overlap = intersection / denominator
        order = remaining[overlap <= threshold]

    return np.array(keep, dtype=np.int64)

def calibrate_boxes(boxes , offsets):
    
    #Tính chiều rộng và chiều cao
    wh = boxes[:,2:4] - boxes[:,0:2]
    scale = np.concatenate([wh , wh], axis = 1)
    
    return boxes + offsets * scale

def square_boxes(boxes):
    wh = boxes[:, 2:4] - boxes[:, 0:2]
    side = np.max(wh, axis=1)

    center = (boxes[:, :2] + boxes[:, 2:4]) / 2

    result = boxes.copy()
    result[:, :2] = center - side[:, None] / 2
    result[:, 2:] = center + side[:, None] / 2

    return result


def valid_box_indices(boxes):
    valid = (
        np.isfinite(boxes).all(axis=1)
        & (boxes[:, 2:] > boxes[:, :2]).all(axis=1)
    )
    return np.flatnonzero(valid)


def decode_landmarks(boxes, points):
    wh = boxes[:, 2:] - boxes[:, :2]

    xs = boxes[:, 0, None] + points[:, :5] * wh[:, 0, None]
    ys = boxes[:, 1, None] + points[:, 5:] * wh[:, 1, None]

    return np.stack([xs, ys], axis=-1).astype(np.float32)