import base64
import string
import urllib.request
import urllib.parse
import uuid
from io import BytesIO
from urllib import parse

import time

import os

import shutil
from PIL import Image
from flask import json

roor_dir = 'D:/workspace/work/crack/temp/'
target_dir = 'D:/workspace/work/crack/test4/'

def readCaptchaText():
    out = open('my.log', encoding='utf-8')
    lines = out.readlines()
    last_line=''
    for line in lines:
        if line.count('识别结果=')>0:
            last_line=line.replace('\n','')
            print(last_line)
    return last_line.split('识别结果=')[1]

def wrap_gen_captcha_img():
    url = 'http://120.194.46.247:20010/edcreg/weChatActivate/getCaptcha'

    data ={'transactionId':'108520180727205100442082'}

    # 携带cookie进行访问
    headers = {
        'Accept':'text/plain, */*; q=0.01',
        'Accept-Encoding':'gzip, deflate',
        'Accept-Language':'zh-CN,zh;q=0.8',
        'Connection':'keep-alive',
        'Content-Length':'38',
        'Content-Type':'application/x-www-form-urlencoded; charset=UTF-8',
        'Cookie':'JSESSIONID=AC276B7D471B4DD7273106387545A74B',
        'Host':'120.194.46.247:20010',
        'Origin':'http://120.194.46.247:20010',
        'Referer':'http://120.194.46.247:20010/edcreg-web/videorealname/realnameActive/realNameActivateM.html',
        'User-Agent':'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.75 Safari/537.36',
        'X-Requested-With':'XMLHttpRequest'
    }
    data = parse.urlencode(data).encode('utf-8')
    predict_code=''
    for i in range(9):
        request = urllib.request.Request(url=url, headers=headers,data=data)
        response = urllib.request.urlopen(request)
        # 输出所有
        content = response.read().decode()
        #print(content)
        jsonData = json.loads(content)
        image_data = jsonData["bean"]['jpgBase64']
        binary_data=base64.b64decode(image_data)
        image=Image.open(BytesIO(binary_data))
        file_name=uuid.uuid1().hex;
        #image.show()
        shutil.rmtree(roor_dir)
        os.mkdir(roor_dir)
        image.save(roor_dir+file_name+".jpg","JPEG")
        time.sleep(0.01)
        p = os.system(os.path.abspath(os.path.join(os.getcwd(), ".."))+"/autoit/predict.exe D:\workspace\work\crack\\temp")
        if predict_code!=readCaptchaText():
            predict_code=readCaptchaText();
            copy_to_target(file_name,predict_code)
        else:
            copy_to_target(file_name)
        time.sleep(1)

def copy_to_target(source_file,predict_code=''):
    if predict_code=='':
        os.remove(roor_dir+"/"+source_file+'.jpg')
    else:
        shutil.copyfile(roor_dir+"/"+source_file+".jpg", target_dir+"/__"+predict_code+'__'+source_file+'.jpg')
        os.remove(roor_dir + "/" + source_file+'.jpg')

if __name__ == "__main__":
    for i in range(500):
        wrap_gen_captcha_img()
        p = os.system(os.path.abspath(os.path.join(os.getcwd(), "..")) + "/autoit/restart.exe D:\workspace\work\captureImgName\captureName.exe")
