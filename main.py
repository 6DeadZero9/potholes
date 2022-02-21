import numpy as np 
import argparse
import json
import cv2
from sqlalchemy import true

def update_trackbars(window_name, trackbars):
    """
        Read values from trackbars
    """

    for trackbar in trackbars:
        trackbars[trackbar]['value'] = cv2.getTrackbarPos(trackbar, window_name)
    
    return trackbars

my_parser = argparse.ArgumentParser(description='')
my_parser.add_argument('-p',
                       '--path',
                       action='store',
                       type=str,
                       required=True,
                       help='Image path')

my_parser.add_argument('-n',
                       '--window_name',
                       action='store',
                       type=str,
                       default='Test',
                       help='Name of the opencv window')

args = my_parser.parse_args()
args_dict = vars(args)

# Read and resize image

image = cv2.imread(args_dict['path'])
image = cv2.resize(image, (420, 300))
cv2.namedWindow(args_dict['window_name'])

# Create ROI 

polygon = np.array([[0, int(image.shape[0] * 0.6)], [int(image.shape[1] * 0.33), int(image.shape[0] * 0.4)],
                    [int(image.shape[1] * 0.67), int(image.shape[0] * 0.4)], [image.shape[1], int(image.shape[0] * 0.8)],
                    [image.shape[1], 0], [0, 0]])

# Extract ROI from image

image = cv2.fillPoly(image, pts = [polygon], color = (0, 0, 0))

# Read trackbar config and create trackbards on given window

with open('config.json', 'r') as config:
    trackbars = json.load(config)

    for trackbar in trackbars:
        cv2.createTrackbar(trackbar, args_dict['window_name'], trackbars[trackbar]['start_value'], trackbars[trackbar]['max'], lambda x: None)

while True:
        # Update trackbar values and create kernel matrix

        trackbars = update_trackbars(args_dict['window_name'], trackbars)
        kernel = (3 + trackbars['kernel']['value'] * 2, 3 + trackbars['kernel']['value'] * 2)

        # Create HSV mask and apply it on image

        lower = np.array([trackbars['HMin']['value'], trackbars['SMin']['value'], trackbars['VMin']['value']])
        upper = np.array([trackbars['HMax']['value'], trackbars['SMax']['value'], trackbars['VMax']['value']])

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(image, image, mask=mask)

        # Convert image to gray and apply adaptive threshold after bluring the image

        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, kernel, 0)
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 3 + trackbars['threshold']['value'] * 2, 2)
        
        gray = cv2.merge((gray, gray, gray))
        combined = np.hstack([image, result, gray])

        cv2.imshow(args_dict['window_name'], combined)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            break

cv2.destroyAllWindows()
