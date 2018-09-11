import numpy as np
from utils import IoU

def _GenRandomBoxes(size=10):
    def _GenRandomBox():
        rand = np.random.randint(low=0, high=101, size=4)
        box = np.array([rand[0], rand[1], rand[0]+rand[2], rand[1]+rand[3]])
        box[box<0]=0
        return box

    arrays = []
    for _ in range(size):
        arrays.append(_GenRandomBox())
    return np.stack(arrays)
    
def _GenPredictionBox(box, fp=False):
    w = box[2] - box[0]
    h = box[3] - box[1]
    dw = int(w*0.7) if fp else int(w*0.3)
    dh = int(h*0.7) if fp else int(h*0.3)
    rand_x = np.random.randint(low=-dw, high=dw+1, size=2)
    rand_y = np.random.randint(low=-dh, high=dh+1, size=2)
    new_box = [box[0]+rand_x[0], box[1]+rand_y[0], box[2]+rand_x[1], box[3]+rand_y[1]]

    new_box = np.array(new_box)
    new_box[new_box<0]=0
    if new_box[2]<new_box[0]: new_box[2]=new_box[0]+1
    if new_box[3]<new_box[1]: new_box[3]=new_box[1]+1

    return new_box

def _GenFileNames(size=100):
    result = []
    while len(result) < size:
        n = np.random.randint(low=0, high=10000)
        s = str(n).zfill(4)
        s+='.jpg'
        if s not in result:
            result.append(s)
    return result
    
def _GenImgData(file_id):
    num_gt = np.random.randint(low=1, high=6)
    gt_boxes = _GenRandomBoxes(size=num_gt)
    gt_key = file_id
    gt_value = gt_boxes

    prediction = []
    for idx in range(num_gt):
        if np.random.random() < 0.2:
            continue
        fp = True if np.random.random()<0.2 else False
        pred_box = _GenPredictionBox(gt_boxes[idx], fp)
        iou = np.max(IoU(pred_box, gt_boxes))
        dc = (np.random.random()-0.5)/3
        confidence = iou + dc
        result = {'bbox':pred_box, 'confidence':confidence, 'file_id':file_id}
        prediction.append(result)
        
    return gt_key, gt_value, prediction, num_gt
    
def GenFakeData(size=10):
    files = _GenFileNames(size=size)
    gts = {}
    predictions = []
    num_gt = 0
    for file in files:
        gt_key, gt_value, prediction, num_gt_this_img = _GenImgData(file)
        gts[gt_key] = gt_value
        predictions += prediction
        num_gt += num_gt_this_img
    return gts, predictions, num_gt