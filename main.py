import numpy as np 
import cv2

def nothing(x):
    pass

def update_trackbars(window_name, trackbars):
    for trackbar in trackbars:
        trackbars[trackbar]['value'] = cv2.getTrackbarPos(trackbar, window_name)
    
    return trackbars

window_name = 'Test'
cv2.namedWindow(window_name)
image = cv2.imread("potholes_5.jpg")
image = cv2.resize(image, (420, 300))

polygon = np.array([[0, int(image.shape[0] * 0.6)], [int(image.shape[1] * 0.33), int(image.shape[0] * 0.4)],
                    [int(image.shape[1] * 0.67), int(image.shape[0] * 0.4)], [image.shape[1], int(image.shape[0] * 0.8)],
                    [image.shape[1], 0], [0, 0]])


image = cv2.fillPoly(image, pts = [polygon], color = (0, 0, 0))

trackbars = {
    'HMin': {
        'max': 179,
        'start_value': 0,
        'value': None
    },
    'SMin': {
        'max': 255,
        'start_value': 2,
        'value': None
    },
    'VMin': {
        'max': 255,
        'start_value': 0,
        'value': None
    },
    'HMax': {
        'max': 179,
        'start_value': 54,
        'value': None
    },
    'SMax': {
        'max': 255,
        'start_value': 105,
        'value': None
    },
    'VMax': {
        'max': 255,
        'start_value': 98,
        'value': None
    },
    'kernel': {
        'max': 20,
        'start_value': 0,
        'value': None
    },
    'threshold': {
        'max': 20,
        'start_value': 10,
        'value': None
    }
}

for trackbar in trackbars:
    cv2.createTrackbar(trackbar, window_name, trackbars[trackbar]['start_value'], trackbars[trackbar]['max'], nothing)


while True:
        trackbars = update_trackbars(window_name, trackbars)
        kernel = (3 + trackbars['kernel']['value'] * 2, 3 + trackbars['kernel']['value'] * 2)

        lower = np.array([trackbars['HMin']['value'], trackbars['SMin']['value'], trackbars['VMin']['value']])
        upper = np.array([trackbars['HMax']['value'], trackbars['SMax']['value'], trackbars['VMax']['value']])

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(image, image, mask=mask)

        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, kernel, 0)
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 3 + trackbars['threshold']['value'] * 2, 2)
        
        gray = cv2.merge((gray, gray, gray))
        combined = np.hstack([image, result, gray])

        cv2.imshow(window_name, combined)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            break

cv2.destroyAllWindows()
