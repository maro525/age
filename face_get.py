# -*- coding: utf-8 -*-
# http://famirror.hateblo.jp/entry/2015/12/19/180000

import cv2
import os
# ファイル操作するモジュール
import shutil


cascade_path = '../../../.pyenv/versions/anaconda3-4.1.1/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml'
cascade = cv2.CascadeClassifier(cascade_path)


def face_recog(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(image_path + '画像が開けません')
        return None
    else:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        facerect = cascade.detectMultiScale(image_gray, scaleFactor=1.2, minNeighbors=2, minSize=(10, 10))
        print(image_path + '開けた')
        return facerect, image

image_path = "image_test/"
out_path = "face_test/"

os.mkdir(out_path)
for f in os.listdir(image_path):
    os.mkdir(out_path + "/" + f)
    print(out_path + "/" + f + " 創った")
    t = 0
    for i in os.listdir(image_path + f):
        facerect, image = face_recog(image_path + f + "/" + i)
        print(facerect)
        for rect in facerect:
            x = rect[0]
            y = rect[1]
            width = rect[2]
            height = rect[3]
            dst = image[y:y + height, x:x + width]
            new_image_path = out_path + f + "/"+ str(i)
            cv2.imwrite(new_image_path, dst)
            print(new_image_path + "置いた")
            t += 1