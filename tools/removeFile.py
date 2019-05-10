#!/usr/bin/python
# -*- coding: UTF-8 -*-


import sys
import os
import shutil
import numpy as np
import cv2

colorNum = 0
infNum = 0
deep = 0

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

def judgeFileexist(pathF,filename):
    road = os.path.join(pathF,filename)
    if os.path.exists(road):
        filename= os.path.splitext(filename)[0]+"_1.bmp"
        return judgeFileexist(pathF,filename)
    else:
        return road



def autoMoveFile(pathA,pathB):#
    if not os.path.exists(pathB):
            os.makedirs(pathB)
    files = os.listdir(pathA)
    for index,name in enumerate(files):
        #if index <11:
            #continue
        print(name)
        loc = chineseMJ.index(name)
        if loc !=-1:
            pathBt = os.path.join(pathB,alphabetMJ[loc])
            if not os.path.exists(pathBt):
                os.mkdir(pathBt)
            pathAt = os.path.join(pathA,name)
            picAlist = os.listdir(pathAt)
            for picname in picAlist:
                Apath = os.path.join(pathAt,picname)
                Bpath = judgeFileexist(pathBt,picname)
                shutil.copyfile(Apath,Bpath)

def autoMovePic(pathA,pathB):
    files = os.listdir(pathA)
    #lenMJ = len(chineseMJ)
    if not os.path.exists(pathB):
            os.makedirs(pathB)
    for index,name in enumerate(files):
        #if index <12000:
            #continue
        if name.find("marre") != -1:
            print("%s,marre%d"%(name,index))
            continue
        remb = -1
        for nl,flod in enumerate(chineseMJ):
            if name.find(flod)!=-1:
                remb = nl
                if name.find('倒') !=-1 and flod.find("倒") == -1 :
                    continue
                else:
                    break
        if remb ==  -1:
            print("notfind%d"%index)
            continue
        apath = os.path.join(pathA,name)
        pathBt = os.path.join(pathB,alphabetMJ[remb])
        
        if not os.path.exists(pathBt):
            os.mkdir(pathBt)
        if len(os.listdir(pathBt))>5000:
            print("%s is than 5000,index:%d"%(alphabetMJ[remb],index))
            continue
        bpath = judgeFileexist(pathBt,name)
        #bpath = os.path.join(pathBt,name)
        shutil.copyfile(apath,bpath)
        print(apath,bpath)
        print ("copy %s,real%d"%(alphabetMJ[remb],index))



def countEveryFile(pathin):
    if not os.path.exists(pathin):
        return
    else:
        files = os.listdir(pathin) 
        for name in files:
            print("%s:"%name)
            print(len(os.listdir(os.path.join(pathin,name))))

            
          
def readYuvFile(filename,width,height):
    fp=open(filename,'rb')
    uv_width=width//2
    uv_height=height//2
   
    Y=np.zeros((height,width),np.uint8,order='C')
    U=np.zeros((uv_height,uv_width),np.uint8,'C')
    V=np.zeros((uv_height,uv_width),np.uint8,'C')
    for m in range(height):
        for n in range(width):
            Y[m,n]=ord(fp.read(1))
    for m in range(uv_height):
        for n in range(uv_width):
            V[m,n]=ord(fp.read(1))
            U[m,n]=ord(fp.read(1))
    fp.close()
    return (Y,U,V)


def yuv2rgb(Y,U,V,width,height):
    U=np.repeat(U,2,0)
    U=np.repeat(U,2,1)
    V=np.repeat(V,2,0)
    V=np.repeat(V,2,1)
    rf=np.zeros((height,width),float,'C')
    gf=np.zeros((height,width),float,'C')
    bf=np.zeros((height,width),float,'C')
    rf=Y+1.14*(V-128.0)
    gf=Y-0.395*(U-128.0)-0.581*(V-128.0)
    bf=Y+2.032*(U-128.0)
    for m in range(height):
        for n in range(width):
            if(rf[m,n]>255):
                rf[m,n]=255
            if(gf[m,n]>255):
                gf[m,n]=255
            if(bf[m,n]>255):
                bf[m,n]=255
    r=rf.astype(np.uint8)
    g=gf.astype(np.uint8)
    b=bf.astype(np.uint8)
    return (r,g,b)

def  read565rgb(filename,width,height):
    re = 0xf8
    gr0 = 0xe0
    gr1 = 0x07
    bl = 0x1f
    fp=open(filename,'rb')
    R=np.zeros((height,width),np.uint8,order='C')
    G=np.zeros((height,width),np.uint8,'C')
    B=np.zeros((height,width),np.uint8,'C')
    for m in range(height):
        for n in range(width):
           data0 = ord(fp.read(1))
           data1 = ord(fp.read(1))
           B[m,n] = (data0&bl)<<3
           gn0 = (data0&gr0)>>3
           gn1 = (data1&gr1)<<5
           G[m,n]=gn0+gn1
           R[m,n]=(data1&re)
    fp.close()
    return (R,G,B)

def changesuffix(path):
    for parent, dirnames, filenames in os.walk(path):
        for filename in filenames:
            picname,suffixN = os.path.splitext(filename)
            if  suffixN==".bmp_1":
                realn = picname+".bmp"
                Bpath = judgeFileexist(parent,realn)
                Apath = os.path.join(parent,filename)
                os.rename(Apath,Bpath)
                #shutil.copyfile(Apath,Bpath)
           

if __name__ == "__main__":
    #needmove = 'H:\\facedetection\\faceData\\colorAndInf'
    #needmove='R:\\pengtao\\X1\\picdata\\train0'
    #tomove = 'R:\\pengtao\\X1\\picdata\\TrainCard'
    #needmove='/datasdb/pengtao/X1/picdata/train0'
    #needmove='/datasdb/pengtao/X1/picdata/binaryPic/a83'
    #needmove='/datasdb/pengtao/X1/picdata/binaryPic/normal/testfold'
    #needmove='/datasdb/pengtao/X1/picdata/binaryPic/33'
    #needmove='/datasdb/pengtao/X1/picdata/binaryPic/42'
    #needmove= 'R:\\pengtao\\X1\\picdata\\binaryPic\\big'
    #needmove= 'R:\\pengtao\\X1\\picdata\\binaryPic\\new'
    needmove= 'R:\\pengtao\\X1\\picdata\\binaryPic\\x1f'
    #tomove = '/datasdb/pengtao/X1/picdata/TrainCard'
    tomove = 'G:/X1_CNN/TrainCard'
    #autoMoveFile(needmove,tomove)
    #countEveryFile(tomove)
    
    autoMovePic(needmove,tomove)

    #changesuffix(tomove)

    #test= 'I:\\HBB\\faceDetTest\\00000003\\室内\\color_9\\ID_9_frame_1_colours.bin'

    #(r,g,b) = read565rgb(test, 240,320)
    #(y,u,v) = readYuvFile(test,240,320)
    #cv2.imshow('tde',y)
    #cv2.waitKey(0)
    #(r,g,b)=  yuv2rgb(y,u,v,240,320)
    '''
    #tesre ='I:\\HBB\\faceDetTest\\00000003\\室内\\color_9\\ID_9_frame_1_colours.bmp'
    tes ='H:\\facedetection\\faceData\colorandinfr\\ID_9_frame_1_colours.bmp'
    reaimg = cv2.imread(tes)
   
    b0 = np.zeros((reaimg.shape[0],reaimg.shape[1]),dtype=reaimg.dtype)
    g0 = np.zeros((reaimg.shape[0],reaimg.shape[1]),dtype=reaimg.dtype)
    r0 = np.zeros((reaimg.shape[0],reaimg.shape[1]),dtype=reaimg.dtype)
    b0[:,:] = reaimg[:,:,0]  # 复制 b 通道的数据
    g0[:,:] = reaimg[:,:,1]  # 复制 g 通道的数据
    r0[:,:] = reaimg[:,:,2]  # 复制 r 通道的数据
    cv2.imshow("Blue0",b0)
    cv2.imshow("Red0",r0)
    cv2.imshow("Green0",g0)
    
    img = cv2.merge([b,g,r])
    zeros = np.zeros(img.shape[:2], dtype = "uint8")
    
    cv2.imshow('ter',r)
    cv2.imshow('teb',b)
    cv2.imshow('teg',g)
    cv2.imshow("Blue", cv2.merge([b, g, r]))
    #cv2.imshow("Green", cv2.merge([ r,g,b]))
    #cv2.imshow("Red", cv2.merge([g,b, r]))
    #cv2.waitKey(0)
  
    #cv2.imshow('te',img)
    cv2.waitKey(0)
    '''
    




