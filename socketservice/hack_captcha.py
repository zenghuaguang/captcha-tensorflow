"""
网络结构
"""
import base64

import io
import tensorflow as tf
import time
from PIL import Image
import numpy as np

from capt.cfg import IMAGE_HEIGHT, IMAGE_WIDTH, CHAR_SET_LEN, MAX_CAPTCHA, model_path
from capt.utils import vec2text, convert2gray

X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN])
keep_prob = tf.placeholder(tf.float32)  # dropout

def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):

    """
    定义CNN
    cnn在图像大小是2的倍数时性能最高, 如果你用的图像大小不是2的倍数，可以在图像边缘补无用像素。
    np.pad(image,((2,3),(2,2)), 'constant', constant_values=(255,))  # 在图像上补2行，下补3行，左补2行，右补2行
    """

    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    # w_c1_alpha = np.sqrt(2.0/(IMAGE_HEIGHT*IMAGE_WIDTH)) #
    # w_c2_alpha = np.sqrt(2.0/(3*3*32))
    # w_c3_alpha = np.sqrt(2.0/(3*3*64))
    # w_d1_alpha = np.sqrt(2.0/(8*32*64))
    # out_alpha = np.sqrt(2.0/1024)

    # 3 conv layer
    #with tf.name_scope('conv1') as scope:
    w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32]))
    b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob)

    #with tf.name_scope('conv2') as scope:
    w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
    b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)

    #with tf.name_scope('conv3') as scope:
    w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
    b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)

    # Fully connected layer
    w_d = tf.Variable(w_alpha * tf.random_normal([8 * 20 * 64, 1024]))
    b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
    dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)

    w_out = tf.Variable(w_alpha * tf.random_normal([1024, MAX_CAPTCHA * CHAR_SET_LEN]))
    b_out = tf.Variable(b_alpha * tf.random_normal([MAX_CAPTCHA * CHAR_SET_LEN]))
    out = tf.add(tf.matmul(dense, w_out), b_out)  # 36*4
    # out = tf.reshape(out, (CHAR_SET_LEN, MAX_CAPTCHA))  # 重新变成4,36的形状
    # out = tf.nn.softmax(out)
    return out

def stringToRGB(base64_str):
    base64_str=str(base64_str)
    if base64_str.split("base64,").__len__()>1:
        base64_str=base64_str.split("base64,")[1]
    imgdata = base64.b64decode(str(base64_str))
    captcha_image = Image.open(io.BytesIO(imgdata))
    #captcha_image.show()
    #captcha_image = captcha_image.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.ANTIALIAS)
    # # 如果验证码图片过大,压缩图片
    if captcha_image.height > IMAGE_HEIGHT or captcha_image.width > IMAGE_WIDTH:
        captcha_image = captcha_image.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.ANTIALIAS)
    captcha_image = np.array(captcha_image)
    # # 如果验证码图片过小,在图像边缘补无用像素
    if captcha_image.shape[0] < IMAGE_HEIGHT or captcha_image.shape[1] < IMAGE_WIDTH:
        top_width = round((IMAGE_HEIGHT - captcha_image.shape[0]) / 2)
        down_width = IMAGE_HEIGHT - captcha_image.shape[0] - top_width
        left_width = round((IMAGE_WIDTH - captcha_image.shape[1]) / 2)
        right_width = IMAGE_WIDTH - captcha_image.shape[1] - left_width
        # cnn在图像大小是2的倍数时性能最高, 如果你用的图像大小不是2的倍数，可以在图像边缘补无用像素。
        captcha_image = np.pad(captcha_image, ((top_width, down_width), (left_width, right_width), (0, 0)), 'constant',
                               constant_values=(255,))  # 在图像上补2行，下补3行，左补2行，右补2行
    return  captcha_image

def hack_function(sess, predict, captcha_image):
    """
    装载完成识别内容后，
    :param sess:
    :param predict:
    :param captcha_image:
    :return:
    """
    text_list = sess.run(predict, feed_dict={X: [captcha_image], keep_prob: 1})

    text = text_list[0].tolist()
    vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
    i = 0
    for n in text:
        vector[i * CHAR_SET_LEN + n] = 1
        i += 1
    return vec2text(vector)

def hack_captcha(captcha_image):
    # 定义预测计算图
    output = crack_captcha_cnn()

    predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    saver = tf.train.Saver()
    predict_text="error"
    with tf.Session() as sess:
        # saver = tf.train.import_meta_graph(save_model + ".meta")
        saver.restore(sess, tf.train.latest_checkpoint(model_path))
        stime = time.time()
        image = convert2gray(captcha_image)
        image = image.flatten() / 255
        predict_text = hack_function(sess, predict, image)
    return predict_text

if __name__ == "__main__":
    base64_str="data:image/jpeg;base64,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"
    iamge_array = stringToRGB(base64_str)
    predict_text =hack_captcha(iamge_array)
    print(predict_text)