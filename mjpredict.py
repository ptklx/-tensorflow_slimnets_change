#coding=utf-8

import tensorflow as tf 
import numpy as np 
import pdb
import cv2
import os
import glob
import sys
sys.path.append("./slim")


import slim.nets.inception_v3 as inception_v3
import slim.nets.resnet_v2 as resnet_v2
#from create_tf_record import *
import tensorflow.contrib.slim as slim




chineseMJ= [
	"一万", "二万", "三万", "四万", "五万", "六万", "七万", "八万", "九万", 
    "一筒", "二筒", "三筒", "四筒", "五筒", "六筒", "七筒", "八筒", "九筒",
    "一条", "二条", "三条", "四条", "五条", "六条", "七条", "八条", "九条",
    "东风", "南风", "西风", "北风", "红中", "发财", "白板", 
    "春", "夏", "秋", "冬", "梅", "兰", "菊", "竹",
	"一万倒", "二万倒", "三万倒", "四万倒", "五万倒", "六万倒", "七万倒", "八万倒", "九万倒",
	"六筒倒", "七筒倒", 
	"一条倒", "三条倒", "七条倒", 
	"东风倒", "南风倒", "西风倒", "北风倒", "红中倒", "发财倒", 
	"春倒", "夏倒", "秋倒", "冬倒", "梅倒", "兰倒", "菊倒", "竹倒", "百搭", "百搭倒", "纯白" ]


alphabetMJ=[
    "yiwan",  "erwan",  "sanwan",  "siwan",  "wuwan",  "liuwan",  "qiwan",  "bawan",   "jiuwan", 
    "yitong", "ertong", "santong", "sitong", "wutong", "liutong",  "qitong", "batong", "jiutong",
    "yitiao", "ertiao", "santiao", "sitiao", "wutiao", "liutiao", "qitiao", "batiao", "jiutiao",
    "dongfeng", "nanfeng", "xifeng", "beifeng","hongzhong", "facai", "baiban",
    "chun", "xia", "qiu", "dong", "mei", "lan", "ju", "zhu", 
    "yiwandao", "erwandao", "sanwandao", "siwandao", "wuwandao", "liuwandao", "qiwandao", "bawandao","jiuwandao",
    "liutongdao", "qitongdao", 
    "yitiaodao", "santiaodao", "qitiaodao",
    "dongfengdao", "nanfengdao", "xifengdao", "beifengdao", "hongzhongdao", "facaidao",
    "chundao", "xiadao", "qiudao", "dongdao", "meidao", "landao","judao", "zhudao", "baida", "baidadao", "guangpai"]


# -*- coding: utf-8 -*-
import cv2
import numpy as np
 
## 读取图像，解决imread不能读取中文路径的问题
def cv_imread(filePath):
    cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    ## imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
    #cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
    return cv_img

    ##这样是保存到了和当前运行目录下
    #cv2.imencode('.jpg', img)[1].tofile('百合.jpg')




def read_image(filename, resize_height, resize_width,normalization=False):
    '''
    读取图片数据,默认返回的是uint8,[0,255]
    :param filename:
    :param resize_height:
    :param resize_width:
    :param normalization:是否归一化到[0.,1.0]
    :return: 返回的图片数据
    '''

    #bgr_image = cv2.imread(filename)
    bgr_image= cv_imread(filename)
    cv2.imshow('dg',bgr_image)

    cv2.waitKey(50)
    if len(bgr_image.shape)==2:#若是灰度图则转为三通道
        print("Warning:gray image",filename)
        bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2BGR)

    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)#将BGR转为RGB
    # show_image(filename,rgb_image)
    # rgb_image=Image.open(filename)

    if resize_height>0 and resize_width>0:
        rgb_image=cv2.resize(rgb_image,(resize_width,resize_height))
    rgb_image=np.asanyarray(rgb_image)
    if normalization:
        # 不能写成:rgb_image=rgb_image/255
        rgb_image=rgb_image/255.0
    # show_image("src resize image",image)
    return rgb_image

def  predict(models_path,image_dir,labels_nums, data_format,modelselect=0):
    [batch_size, resize_height, resize_width, depths] = data_format

    #labels = np.loadtxt(labels_filename, str, delimiter='\t')
    input_images = tf.placeholder(dtype=tf.float32, shape=[None, resize_height, resize_width, depths], name='input')
    if modelselect == 0:
        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            out, end_points = inception_v3.inception_v3(inputs=input_images, num_classes=labels_nums, dropout_keep_prob=1.0, is_training=False)
    elif modelselect == 1:
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            out, end_points = resnet_v2.resnet_v2_101(inputs=input_images, num_classes=labels_nums, is_training=False)


    # 将输出结果进行softmax分布,再求最大概率所属类别
    score = tf.nn.softmax(out,name='pre')
    class_id = tf.argmax(score, 1)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, models_path)
    images_list=glob.glob(os.path.join(image_dir,'*.bmp'))
    for image_path in images_list:
        im=read_image(image_path,resize_height,resize_width,normalization=True)
        im=im[np.newaxis,:]
        #pred = sess.run(f_cls, feed_dict={x:im, keep_prob:1.0})
        pre_score,pre_label = sess.run([score,class_id], feed_dict={input_images:im})
        max_score=pre_score[0,pre_label]
        print ("{} is: pre labels:{},name:{} score: {}".format(image_path,pre_label,chineseMJ[pre_label[0]], max_score))
        
    sess.close()


if __name__ == '__main__':

    class_nums=73
    image_dir='R:\\pengtao\\X1\\picdata\\marre'
    labels_filename='dataset/label.txt'

    modelselect = 1     # 0 inception  1 resnet_v2
    if modelselect==0:
        models_path='./mjmodels/inceptionV3_2/model.ckpt-10000'
        batch_size = 1  #
        resize_height = 299  # 指定存储图片高度
        resize_width = 299  # 指定存储图片宽度
        depths=3
    elif modelselect ==1:
        models_path='./mjmodels/resnet_v2/model.ckpt-10000'
        batch_size = 1  #
        resize_height = 224  # 指定存储图片高度
        resize_width = 224  # 指定存储图片宽度
        depths=3

    data_format=[batch_size,resize_height,resize_width,depths]
    predict(models_path,image_dir,  class_nums, data_format,modelselect)
