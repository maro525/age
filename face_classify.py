# coding: utf-8
import os
import cv2
import numpy as np
import tensorflow as tf

'''
関数
'''


def img_to_matrix(filename, verbose=False):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (28, 28))
    img_array = np.array(img)
    return img_array


def flatten_matrix(img):
    s = img.shape[0] * img.shape[1]
    img_wide = img.reshape(1, s)
    return img_wide


def tensorflow(train_x, train_y, test_x, test_y):
    # 学習したいモデルを記述
    # 入力変数と出力変数のプレースホルダを生成
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 6])
    # モデルパラメータ
    W1 = tf.Variable(tf.truncated_normal([784, 100]))
    b1 = tf.Variable(tf.zeros([100]))
    W2 = tf.Variable(tf.truncated_normal([100, 6]))
    b2 = tf.Variable(tf.zeros([6]))
    # モデル式
    h = tf.sigmoid(tf.matmul(x, W1) + b1)
    u = tf.matmul(h, W2) + b2
    y = tf.nn.softmax(u)

    # 学習やテストの関数
    # 調査関数
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(u, y_))
    # 最適手段化(最急降下法)
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    # 正答率
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 実際に学習処理を実行する
    # セッションを準備し、変数を初期化
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    # バッチ型確率的勾配降下法でパラメータ更新
    for i in range(5000):
        batch_xs = train_x
        batch_ys = train_y
        # 学習
        _, l = sess.run([train_step, loss], feed_dict={x: batch_xs, y_: batch_ys})
        if (i + 1) % 1000 == 0:
            print("step=%3d, loss=%.2f" % (i + 1, l))

    # テスト用データに対して予測子、性能を確認
    new_x = test_x;
    new_y_ = test_y;
    # 予測と性能評価
    accuracy, new_y = sess.run([acc, y], feed_dict={x: new_x, y_: new_y_})
    print("Accuracy (for test data): %6.2f%%" %(accuracy*100))
    print("True Label:", ((np.argmax(new_y_[0:36,], 1)+1) *10))
    print("Est Label :", ((np.argmax(new_y[0:36, ], 1)+1) *10))

    sess.close()


'''
メイン
'''
# 訓練データ成形
img_train_path = "face_train/"
train_data = []
train_labels = []
for f in os.listdir(img_train_path):
    for i in os.listdir(img_train_path + f):
        img = img_to_matrix(img_train_path + f + "/" + i)
        img = flatten_matrix(img)
        train_data.append(img[0])
        train_label = np.zeros(6)
        train_label[int(int(f)/10 -1)] = 1
        train_labels.append(train_label)
train_data = np.array(train_data)
train_labels = np.array(train_labels)
print("train data shape : ", train_data.shape)
print("train labels shape : ", train_labels.shape)

# テストデータ成形
img_test_path = "face_test/"
test_data = []
test_labels = []
for f in os.listdir(img_test_path):
    for i in os.listdir(img_test_path + f):
        img = img_to_matrix(img_test_path + f + "/" + i)
        img = flatten_matrix(img)
        test_data.append(img[0])
        test_label = np.zeros(6)
        test_label[int(int(f)/10 - 1)] = 1
        test_labels.append(test_label)
test_data = np.array(test_data)
test_labels = np.array(test_labels)
print("test data shape : ", test_data.shape)
print("test data shape : ", test_labels.shape)

# 学習
tensorflow(train_data, train_labels, test_data, test_labels)