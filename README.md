## Some important concepts
### Intersection over Union (IoU) 
Intersection over Union is simply a ratio. In the numerator we compute the area of overlap between the predicted bounding box and the ground-truth bounding box. The denominator is the area of union, or more simply, the area encompassed by both the predicted bounding box and the ground-truth bounding box.
![Alt text](https://github.com/Ao-Lee/AveragePrecision/raw/master/figures/iou.jpg)
In the above figure I have included examples of good and bad IoU scores. As you can see, predicted bounding boxes that heavily overlap with the ground-truth bounding boxes have higher scores than those with less overlap. This makes Intersection over Union an excellent metric for evaluating custom object detectors.
![Alt text](https://github.com/Ao-Lee/AveragePrecision/raw/master/figures/different_iou.jpg)

### TP (true positive):
Correctly predicted positive examples. In the object detection context, it means a correct detection. The IOU between the detected box and any ground truth is greater than the threshold (in the PASCAL VOC Challenge, this threshold is set to 0.5. In this library, APIs are provided to change this threshold)

### FP (false positive):
Wrongly predicted negative examples. In the object detection context, it means a wrong detection. The IOU between the detected box and all ground truth bounding boxes is less than the threshold.

### FN (false negative): 
Wrongly predicted positive examples. In the object detection context, it means a ground truth object is not detected

### TN (true negative):
Correctly predicted negative examples. In the object detection context, it means non-object bounding boxes which are not detected by the model. Since there are infinite possible non-object bounding boxes in an image, TN is never used in object detection.



### Precision 
Precision = TP / (TP + FP) = TP / number of detected bounding boxes

### Recall
Recall = TP / (TP + FN) = TP / number of ground truth bounding boxes

## How is Average Precision (AP) computed?
Computing AP for a particular object detection pipeline is essentially a three step process:
1. Compute all the precisions which the confidence of the model varies.
2. Compute all the recalls which the confidence of the model varies.
3. Draw a precision-recall curve for all precisions and recalls, AP is the area under this Precision-Recall curve. In the implementation, an Interpolated precision-recall curve is used, as shown in the figure.
![Alt text](https://github.com/Ao-Lee/AveragePrecision/raw/master/figures/precision_recall_curve.jpg)

## APIs
in order to compute average protection, you need to provoide three piece of information, the ground truth Information, the predicton information and the total number of ground truth objects.
### Ground truth Information
a dictionary of files, representing the ground truth information. Each dictionary contains an image id and ground truth bounding boxes of that image, for example:
```python
{'1.jpg':array([[95, 95, 188, 176]]), '2.jpg': array([[21, 94, 104, 157], [ 9, 88, 56, 178]]}
```                
### Prediction Information
a list, each element in the list representing a prediction, for example:
```python
[
	{'bbox': array([ 83, 73, 156, 101]), 'confidence': 0.53, 'file_id': '0969.jpg'},
	{'bbox': array([ 30, 71, 122, 98]), 'confidence': 0.83, 'file_id': '1440.jpg'}
]
```
### Total number of ground truth objects
The total number of ground truth objects must be provided to correctly compute recall, as discussed above.

## How to run the example?
Run example.py, this is a complete example. Note that it is able to adjust the IOU threshold.

## Common questions:
* __What if many detections have high IOUs to the same ground truth object? Are those detections all considered as true positive?__
No. In this case, only one detection is counted as TP, other detections are false positives

* __I have a multiclass detection scenario, and how to compute MAP among all classes?__
provoide the ground truth, the predicton and the total number of ground truth objects for each paticular class, compute AP for each class, and average the results.

