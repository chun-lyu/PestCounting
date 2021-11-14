## all code
import os
from flask import Flask, jsonify, render_template, request, url_for, send_from_directory
from werkzeug.utils import secure_filename
import math
import cv2
import numpy as np
from PIL import Image, ImageOps
from flask_cors import CORS
import requests
import base64


## sticky card crop function
def resizeImg(image, height=900):
    h, w = image.shape[:2]
    pro = height / h
    size = (int(w * pro), int(height))
    img = cv2.resize(image, size)
    return img

# canny edge detection
def getCanny(image):
    # 高斯模糊
    binary = cv2.GaussianBlur(image, (11, 11), 2, 2)
    # 边缘检测
    binary = cv2.Canny(binary, 60, 240, apertureSize=3)
    # 膨胀操作 尽量使边缘闭合
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)
    return binary

# findMaxContour
def findMaxContour(image):
    # 寻找边缘
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # 计算面积
    max_area = 0.0
    max_contour = []
    for contour in contours:
        currentArea = cv2.contourArea(contour)
        if currentArea > max_area:
            max_area = currentArea
            max_contour = contour
    return max_contour, max_area

# get hulk point of parallel
def getBoxPoint(contour):
    # 多边形拟合凸包
    hull = cv2.convexHull(contour)
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    approx = approx.reshape((len(approx), 2))
    return approx

# get point value in oringal img
def adaPoint(box, pro):
    box_pro = box
    if pro != 1.0 :
        box_pro = box/pro
    box_pro = np.trunc(box_pro)
    return box_pro

# ordering point
def orderPoints(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)# sum of point value(value_x + value_y)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)# diff between value_y and value_x
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

# calculate point distance
def pointDistance(a, b):
    return int(np.sqrt(np.sum(np.square(a - b))))

# PerspectiveTransform
def warpImage(image, box):
    w, h = pointDistance(box[0], box[1]), \
           pointDistance(box[1], box[2])
    dst_rect = np.array([[0, 0],
                         [w - 1, 0],
                         [w - 1, h - 1],
                         [0, h - 1]], dtype='float32')
    M = cv2.getPerspectiveTransform(box, dst_rect)
    warped = cv2.warpPerspective(image, M, (w, h))
    return warped

def sticky_card_crop(path):
    image = cv2.imread(path)
    ratio = 900 / image.shape[0]
    img = resizeImg(image)
    binary_img = getCanny(img)
    max_contour, _ = findMaxContour(binary_img)
    boxes = getBoxPoint(max_contour)
    boxes = adaPoint(boxes, ratio)
    boxes = orderPoints(boxes)
    warped = warpImage(image, boxes)
    return warped

## pading pic
def padding_pic(img):
    img_row_num = 3 - img.shape[0]%3
    img=cv2.copyMakeBorder(img, 0, img_row_num, 0, 0, cv2.BORDER_CONSTANT, value=0)
    return img

## K-Means, Binarization and Elliptical Fit Code
## K-means Code
def pest_count_by_kmeans_and_binarz(Img, num_clusters=2):
    #Img = cv2.imread(path, 1)
    Img = Img[:,:,0]
    # cv2.GaussianBlur
    blur = cv2.GaussianBlur(Img, (9, 9), 0)
    
    # binarization
    _, otsu = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY)
    
    # k-means
    data = otsu.reshape((-1,3))
    data = np.float32(data)
    # stop criteria: criteria flag(means one of them), iteration, epsilon
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # cv2.kmeans input six parameter: source data, cluster number, preset labels, 
    # stop criteria, repeat times, initial center
    # cv2.kmeans return three value: retrurn value type, label of pixel, cluster center
    _,label,_=cv2.kmeans(data, num_clusters, None, criteria, 
                           num_clusters, cv2.KMEANS_RANDOM_CENTERS)
    color = np.uint8([[255, 0, 0],[128, 128, 128]])
    res = color[label.flatten()]
    result = res.reshape((Img.shape))
    return result

## Elliptical Fit Code
def Elliptical_Fit(kmeans_result, original_file):
    blur = cv2.GaussianBlur(kmeans_result, (9, 9), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.Canny input three value: source img, min value, max value
    binary = cv2.Canny(otsu, 80, 80 * 2)
    # cv2.findContours input three values: source img, detect model(external outline only), output value store type 
    contours,_ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #src = original_file
    ellipse_area = list()
    for c in range(len(contours)):
        if contours[c].size/2 >4:
            # 椭圆拟合
            (cx, cy), (a, b), angle = cv2.fitEllipse(contours[c])
            # 绘制椭圆
            if (cx > 0) & (cy > 0):
                cv2.ellipse(original_file, (np.int32(cx), np.int32(cy)),
                           (np.int32(a/2), np.int32(b/2)), angle, 0, 360, (0, 0, 255), 1, 8)
                ellipse_area.append(round(math.pi*a*b/4,3))
    
    #cv2.imwrite("Keamns_Elliptical_Fit_Output.jpg",original_file)
    #src = cv2.cvtColor(original_file, cv2.COLOR_BGR2RGB)
    #plt.imshow(src),plt.xticks([]),plt.yticks([]),plt.show()
    return (original_file, ellipse_area)


## merge all code
def Thrips_Counting(path):
    croped_img = sticky_card_crop(path)
    img = padding_pic(croped_img)
    kmeans_result = pest_count_by_kmeans_and_binarz(img)
    Elliptical_Fit_result, ellipse_area = Elliptical_Fit(kmeans_result, croped_img)
    return Elliptical_Fit_result, ellipse_area

