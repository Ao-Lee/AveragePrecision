from example_data_generator import GenFakeData
from ap import AveragePrecisionOnImages

gts, predictions, num_gt = GenFakeData(size=20)
ap, _, _ = AveragePrecisionOnImages(gts, predictions, num_gt, min_overlap=0.5, validate_input=True)
print(ap)

