import numpy as np

'''
Compute IoU between detect box and ground truth boxes

Parameters:
    box:            detection box, numpy array, shape (4,): x1, y1, x2, y2
    
    boxes:          ground truth boxes, shape (n, 4): x1, y1, x2, y2

Returns:
    iou:            numpy array, shape (n, )
'''
def IoU(box, boxes):
    
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])
    
    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    inter = w * h
    iou = inter / (box_area + area - inter)
    return iou