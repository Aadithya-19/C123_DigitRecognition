import cv2

import pandas as pd

import os, ssl

import numpy as np

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from PIL import Image

import PIL.ImageOps

from sklearn.datasets import fetch_openml

X, y = fetch_openml('mnist_784', version = 1, return_X_y = True)

classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

nclasses = len(classes)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 9, train_size = 7500, test_size = 2500)

X_train_scale = (X_train/255.0)

X_test_scale = (X_test/255.0)

clf = LogisticRegression(solver = 'saga', multi_class = 'multinomial').fit(X_train_scale, y_train)


y_pred = clf.predict(X_test_scale)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

cap = cv2.VideoCapture(0)

while(True):
    try:
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        height, width = gray.shape

        upper_left = (int(width/2-56), int(height/2-56))

        bottom_right = (int(width/2+56), int(height/2+56))

        cv2.rectange(gray, upper_left, bottom_right,(0, 255, 0, 2))

        roi = gray[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]

        ImagePIL = Image.fromarray(roi)

        imgBAW = ImagePIL.convert(L)

        let_imgBAW = PIL.ImageOps.invert(imgBAW)

        pixel_filter = 20

        min_pixel = np.percentile(let_imgBAW, pixel_filter)

        scales_img = np.clip(let_imgBAW-min_pixel, 0, 255)

        max_pixel = np.max(let_imgBAW)

        scales_img = np.asarray(let_imgBAW, 0, 255)/max_pixel

        test_sample = np.array(scales_img.reashape(1, 780))
        
        test_pred = clf.predict(test_sample)
        
        print(test_pred)

cap.release()
cv2.destroyAllWindows()