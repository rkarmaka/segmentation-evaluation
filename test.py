# test the metrics package

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
from metrics.evaluate import evaluate_segmentation
from data.synthetic_cases import *
from rich import print


# load the ground truth and predicted label images
# circle_gt = create_circle_mask((100, 100), 20)
# circle_pred = create_circle_mask((100, 100), 20)

circle_gt = create_paired_circles((100, 100), (20, 30))
circle_pred = create_paired_circles((100, 100), (10, 20))

# evaluate the metrics
results = evaluate_segmentation(circle_gt, circle_pred)
print(results)