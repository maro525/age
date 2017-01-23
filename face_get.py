# -*- coding: utf-8 -*-
import cv2
import os

# このパスは環境に合わせて設定
cascade_path = '../../../.pyenv/versions/anaconda3-4.1.1/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml'
cascade = cv2.CascadeClassifier(cascade_path)

# 顔認識関数
def face_recog(image_path):
    # 画像読み込み
    image = cv2.imread(image_path)
    if image is None:
        print(image_path + '画像が開けません')
        return None
    else:
        # グレースケール変換
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        facerect = cascade.detectMultiScale(image_gray, scaleFactor=1.2, minNeighbors=2, minSize=(10, 10))
        return facerect, image

image_path = "image_test/"
out_path = "face_test/"

os.mkdir(out_path)
for f in os.listdir(image_path):
    os.mkdir(out_path + "/" + f)
    t = 0
    for i in os.listdir(image_path + f):
        # 顔認識する
        facerect, image = face_recog(image_path + f + "/" + i)
        for rect in facerect:
            x = rect[0]
            y = rect[1]
            width = rect[2]
            height = rect[3]
            dst = image[y:y + height, x:x + width]
            new_image_path = out_path + f + "/"+ str(i)
            cv2.imwrite(new_image_path, dst)
            t += 1