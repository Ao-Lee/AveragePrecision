import numpy as np
from utils import IoU

'''
Compute average precision given each result of prediction and the confidence

Parameters:
    prediction:     a list of dictionary. 
                    for example [{'TP':True, 'confidence':0.65}, {'TP':False, 'confidence':0.4}]
                    set TP to True if the prediction is true positive
                    set TP to False if the prediction is false positive
                    
    num_gt_example: int, number of ground truth examples
    
    validate:       bool, turn it on to check the inputs

Returns:
    ap:             int, average precision. note that we assume the inputs belongs to the same class
                    it is possible to utilize this functionality to compute mean average precision via multiple classes
                    
    recall:         recalls computed over different confidence threshold, numpy array of shape (N,)
    
    precision:      precision computed over different confidence threshold, numpy array of shape (N,) 
    
'''
def VocAveragePrecision(prediction, num_gt_example, validate=False):
    def CheckInput(prediction, num_gt_example):
        num_true_positive = 0
        for d in prediction:
            keys = list(d.keys())
            assert 'TP' in keys and 'confidence' in keys and type(d['TP']) is bool
            assert len(keys) == 2
            if d['TP']: num_true_positive +=1
        assert num_true_positive <= num_gt_example
        return
    
    if validate: CheckInput(prediction, num_gt_example)
    
    prediction.sort(key=lambda x:x['confidence'], reverse=True)
    prediction = [info['TP'] for info in prediction]
    prediction = np.array(prediction)
    
    tp = (prediction==True).astype(np.int)
    fp = (prediction==False).astype(np.int)
    
    cumulative_tp = np.cumsum(tp)
    cumulative_fp = np.cumsum(fp)
    
    # precision_scores = cumulative_tp / (np.arange(len(prediction)) + 1)
    precision = cumulative_tp / (cumulative_tp + cumulative_fp)
    recall = cumulative_tp / num_gt_example
    
    '''
    metric      rank(confidence)    extream_value
    precision   inf                 1
    precision   -inf                0
    recall      inf                 0
    recall      -inf                1
    '''
    precision = np.array([1.0] + list(precision) + [0.0])
    recall = np.array([0.0] + list(recall) + [1.0])
    
    for i in range(len(precision)-2, -1, -1):
        precision[i] = max(precision[i], precision[i+1])

    i_list = []
    for i in range(1, len(recall)):
        if recall[i] != recall[i-1]:
            i_list.append(i)
    
    ap = 0.0
    for i in i_list:
        ap += ((recall[i]-recall[i-1])*precision[i])
    return ap, recall, precision
   
'''
Compute average precision given predictions of images

Parameters:     
    gt_info:        a dictionary of files, representing the ground truth infomation 
                    of each image, for example
                        {'1.jpg':bboxes_01, '2.jpg':bboxes_02}
                    each key represents each image file, and the bboxes_01 is 
                    the numpy array of shape (K, 4), representing K ground truth
                    objects.
                    
    predictions_info:
                    a list, each element in the list representing a prediction, for example: [prediction01, prediction02]
                    each prediction is a dictionary, for example
                        {'bbox': array([165., 214., 230., 255.]), 'confidence':0.85, 'file_id':'2.jpg'}
                    bbox is numpy array of shape (4, )
                    file_id should be consistent with the files in gt_info
                    
    num_gt_example: int, number of ground truth examples
    
    validate_input: bool, turn it on to check the inputs
    
    min_overlap:    should be in (0, 1) default value is 0.5 (defined in the PASCAL VOC2012 challenge)
                    
Returns:
    ap:             int, average precision. note that we assume the inputs belongs to the same class
                    it is possible to utilize this functionality to compute mean average precision via multiple classes
                    
    recall:         recalls computed over different confidence threshold, numpy array of shape (N,)
    
    precision:      precision computed over different confidence threshold, numpy array of shape (N,) 
    
'''
def AveragePrecisionOnImages(gt_info, predictions_info, num_gt_example, min_overlap=0.5, validate_input=False):
    result = []
    used_info = {}
    # set 'used' tag for each ground truch bbox
    for file_id in gt_info:
        boxes = gt_info[file_id]
        size = boxes.shape[0]
        used = np.zeros(shape=[size]).astype(np.bool)
        used_info[file_id] = used

    for idx, prediction in enumerate(predictions_info):
        file_id = prediction['file_id']
        box_pred = prediction['bbox']

        boxes_gt = gt_info[file_id]
        iou = IoU(box_pred, boxes_gt)
        idx = iou.argmax()
        used = used_info[file_id][idx]
        confidence = prediction['confidence']

        if iou[idx] >= min_overlap and not used:
            result.append({'TP':True, 'confidence':confidence})
            used_info[file_id][idx] = True
        else:
            result.append({'TP':False, 'confidence':confidence})

    ap, recall, precision = VocAveragePrecision(result, num_gt_example, validate=validate_input)
    return ap, recall, precision
    
    
    
    
    