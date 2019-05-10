# -*-coding:utf-8-*-
"""
    @Project: googlenet_classification
    @File   : create_labels_files.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2018-08-11 10:15:28
"""

import os
import os.path
import cv2

MJname=[
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

maxDataNum = 4000  #

def write_txt(content, filename, mode='w'):
    """保存txt数据
    :param content:需要保存的数据,type->list
    :param filename:文件名
    :param mode:读写模式:'w' or 'a'
    :return: void
    """
    with open(filename, mode) as f:
        for line in content:
            str_line = ""
            for col, data in enumerate(line):
                if not col == len(line) - 1:
                    # 以空格作为分隔符
                    str_line = str_line + str(data) + " "
                else:
                    # 每行最后一个数据用换行符“\n”
                    str_line = str_line + str(data) + "\n"
            f.write(str_line)


def get_files_list(dir,train_test=0):
    '''
    实现遍历dir目录下,所有文件(包含子文件夹的文件)
    :param dir:指定文件夹目录
    :return:包含所有文件的列表->list
    '''
    # parent:父目录, filenames:该目录下所有文件夹,filenames:该目录下的文件名
    files_list = []
    Mjspecies = len(MJname)
    
    for parent, dirnames, filenames in os.walk(dir):
        lenNum = len(filenames)
        for filename in filenames:
            # print("parent is: " + parent)
            # print("filename is: " + filename)
            # print(os.path.join(parent, filename))  # 输出rootdir路径下所有文件（包含子文件）信息
            curr_file = parent.split(os.sep)[-1]
           
            for index,name in  enumerate(MJname):
                if curr_file == name:
                    labels = index
            suffixN = os.path.splitext(filename)
            n_idx = 0
            if  suffixN[1]==".bmp" or  suffixN[1] ==".jpg":
               
                if lenNum < maxDataNum:
                    files_list.append([os.path.join(curr_file, filename), labels])

                    picpath = os.path.join(parent, filename)
                    img = cv2.imread(picpath)
                    img_h,img_w,img_c = img.shape

                    addimg_r = cv2.copyMakeBorder(img,0,0,0,3,cv2.BORDER_CONSTANT,value=[0,0,0]) #right  add 3
                    save_file = os.path.join(parent, suffixN[0]+"%s.bmp"%n_idx)
                    cv2.imwrite(save_file, addimg_r)
                    files_list.append([os.path.join(curr_file, suffixN[0]+"%s.bmp"%n_idx), labels])
                    n_idx+=1

                    addimg_l = cv2.copyMakeBorder(img,0,0,3,0,cv2.BORDER_CONSTANT,value=[0,0,0]) #left  add 3
                    save_file = os.path.join(parent, suffixN[0]+"%s.bmp"%n_idx)
                    cv2.imwrite(save_file, addimg_l)
                    files_list.append([os.path.join(curr_file, suffixN[0]+"%s.bmp"%n_idx), labels])
                    n_idx+=1

                    #addimg_u = cv2.copyMakeBorder(img,4,4,0,0,cv2.BORDER_CONSTANT,value=[0,0,0]) #up down  add 3
                    addimg_s = cv2.resize(img,(img_w,img_h+10))
                    save_file = os.path.join(parent, suffixN[0]+"%s.bmp"%n_idx)
                    cv2.imwrite(save_file, addimg_s)
                    files_list.append([os.path.join(curr_file, suffixN[0]+"%s.bmp"%n_idx), labels])
                    n_idx+=1

                else:    
                    files_list.append([os.path.join(curr_file, filename), labels])
    return files_list


def judgeFileexist(pathF,filename):
    road = os.path.join(pathF,filename)
    if os.path.exists(road):
        filename= os.path.splitext(filename)[0]+"_1.bmp"
        return judgeFileexist(pathF,filename)
    else:
        return road

def renamePic(dir):
    files_list = []
    Mjspecies = len(MJname)  
    for parent, dirnames, filenames in os.walk(dir):
        Picnum = 0
        for filename in filenames:
            # print("parent is: " + parent)
            # print("filename is: " + filename)
            # print(os.path.join(parent, filename))  # 输出rootdir路径下所有文件（包含子文件）信息
            #curr_file = parent.split(os.sep)[-1]
            
            suffixN = os.path.splitext(filename)[1]
            
            if  suffixN==".bmp" or  suffixN ==".jpg":
                fullpath = os.path.join(parent,filename)
                postfix = "_.bmp"
                if suffixN =="_.jpg":
                    postfix = "_.jpg"
                newpath = judgeFileexist(parent,str(Picnum)+postfix)
                Picnum+=1
                #shutil.copyfile(fullpath,newpath)
                os.rename(fullpath,newpath)
                
    return 

if __name__ == '__main__':
    renameorlabels = 0

    train_dir = '/datasdb/pengtao/X1/picdata/TrainCard0'
    val_dir = '/datasdb/pengtao/X1/picdata/Val'
    #train_dir ='R:\\pengtao\\X1\\picdata\\TrainCard0'
    if renameorlabels == 1:
        renamePic(train_dir)
        renamePic(val_dir)
        print("rename is OK!")
    else:
        
        #train_txt = '/datasdb/pengtao/X1/picdata/train.txt'
        train_dir = 'R:/pengtao/X1/picdata/TrainCard0'
        train_txt = 'R:/pengtao/X1/picdata/train.txt'
        train_data = get_files_list(train_dir,train_test=1)
        write_txt(train_data, train_txt, mode='w')
      
        #val_txt = '/datasdb/pengtao/X1/picdata/val.txt'

        val_dir='R:/pengtao/X1/picdata/Val'
        val_txt = 'R:/pengtao/X1/picdata/val.txt'
        val_data = get_files_list(val_dir)
        write_txt(val_data, val_txt, mode='w')

