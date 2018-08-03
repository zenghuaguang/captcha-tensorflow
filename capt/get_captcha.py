import random


import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

from os.path import join
import capt
from capt.cfg import gen_char_set, IMAGE_HEIGHT, IMAGE_WIDTH, MAX_CAPTCHA, CHAR_SET_LEN
from capt.utils import convert2gray, text2vec


def get_captcha_text_and_image(rootdir,file_names):
    file_name=random.choice(file_names)
    #print(file_name)
    captcha_text=file_name[2:6]
    captcha_image = Image.open(join(rootdir,file_name))
    # 如果验证码图片过大,压缩图片
    if captcha_image.height >IMAGE_HEIGHT or captcha_image.width >IMAGE_WIDTH:
        captcha_image = captcha_image.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.ANTIALIAS)
    captcha_image = np.array(captcha_image)
    if captcha_image.shape[0]<IMAGE_HEIGHT or captcha_image.shape[1]<IMAGE_WIDTH:
        top_pad=round((IMAGE_HEIGHT- captcha_image.shape[0])/2)
        down_pad=IMAGE_HEIGHT-captcha_image.shape[0]-top_pad
        left_pad=round((IMAGE_WIDTH- captcha_image.shape[1])/2)
        right_pad=IMAGE_WIDTH-captcha_image.shape[1]-left_pad
        captcha_image=np.pad(captcha_image, ( (top_pad, down_pad),(left_pad, right_pad),(0,0)), 'constant', constant_values=(255,))
    #Image.fromarray(captcha_image.astype('uint8')).convert('RGB').show()
    return captcha_text,captcha_image

#返回打码图片内容，文件名
def get_captcha_text_and_image2(rootdir,file_name):
    #print(file_name)
    captcha_image = Image.open(join(rootdir,file_name))
    # 如果验证码图片过大,压缩图片
    if captcha_image.height !=IMAGE_HEIGHT or captcha_image.width != IMAGE_WIDTH:
        captcha_image = captcha_image.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.ANTIALIAS)
    captcha_image = np.array(captcha_image)
    return file_name,captcha_image

def wrap_get_captcha_text_and_image(rootdir,file_names):
    """
    有时生成图像大小不是(60, 160, 3)
    :return:
    """
    while True:
        text, image = get_captcha_text_and_image(rootdir,file_names)
        if image.shape != (IMAGE_HEIGHT, IMAGE_WIDTH, 3):
            continue
        #print(text)
        return text, image

def get_next_batch_hzeng(batch_size=128):
    """
    # 生成一个训练batch
    :param batch_size:
    :return:
    """
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN])
    rootdir =join(capt.cfg.workspace, 'train')
    file_names = []
    for parent, dirnames, filenames in os.walk(rootdir):  # 三个参数：分别返回1.父目录 2.所有文件夹名字（不含路径） 3.所有文件名字
        file_names = filenames

    for i in range(batch_size):
        text, image = wrap_get_captcha_text_and_image(rootdir,file_names)
        image = convert2gray(image)
        # fig, axarr = pylab.plt.subplots(2, 2)
        # axarr[0, 0].imshow(image)
        # pylab.show()
        batch_x[i, :] = image.flatten() / 255  # (image.flatten()-128)/128  mean为0
        batch_y[i, :] = text2vec(text)

    return batch_x, batch_y


def get_next_batch_test_hzeng(batch_size=128):
    """
    # 生成一个训练batch
    :param batch_size:
    :return:
    """
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN])
    rootdir =join(capt.cfg.workspace, 'test')
    file_names = []
    for parent, dirnames, filenames in os.walk(rootdir):  # 三个参数：分别返回1.父目录 2.所有文件夹名字（不含路径） 3.所有文件名字
        file_names = filenames

    for i in range(batch_size):
        text, image = wrap_get_captcha_text_and_image(rootdir,file_names)
        image = convert2gray(image)
        # fig, axarr = pylab.plt.subplots(2, 2)
        # axarr[0, 0].imshow(image)
        # pylab.show()
        batch_x[i, :] = image.flatten() / 255  # (image.flatten()-128)/128  mean为0
        batch_y[i, :] = text2vec(text)

    return batch_x, batch_y

# 使用pyplot显示图像
def show(img_data):
    plt.imshow()
    plt.show()
import operator;

if __name__ == '__main__':
    #get_next_batch_test_hzeng()
    file_names = []
    rootdir =join(capt.cfg.workspace, 'test')
    for parent, dirnames, filenames in os.walk(rootdir):  # 三个参数：分别返回1.父目录 2.所有文件夹名字（不含路径） 3.所有文件名字
        file_names = filenames
    file_dict={}
    for i in range(20):
        for i in range(256):
            #x=random.randint(0,file_names.__len__())
            file_name=random.choice(file_names)
            if (file_name in file_dict.keys()):
                file_dict[file_name]=file_dict[file_name]+1
            else:
                file_dict[file_name] = 1

    print("length:",file_dict.keys().__len__())

    print("按值进行降序排序结果为:",
          sorted(file_dict.items(), key=operator.itemgetter(1), reverse=True));

