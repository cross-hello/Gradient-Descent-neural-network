import glob
import re
import os
import cv2
import numpy as np

def get_digit(s):
    return re.findall('([0-9])_',s)[0]

def activate(s):
    a=np.zeros((10,1))
    #a=np.zeros((10))// for matrix
    a[int(s)]=1
    return a

def get(dir='manual_digit'):
    if '\\' in dir[-1] or '/' in dir[-1]:
        l=glob.glob(dir+'*.png')
    else:
        #l=glob.glob(dir+'/*.png')
        l=glob.glob(os.path.join(dir,'*.png'))
    data=[]
    for a in l:
        img=cv2.imread(a,cv2.IMREAD_GRAYSCALE)
        if img.shape!=(28,28):
            #cv2.resize(img,(28,28),img,interpolation=)
            img=cv2.resize(img,(28,28))
        img=np.reshape(img,(28*28,1))
        img=1-img/255
        #img=np.reshape(img,(28*28))// for matrix
        #data.append((img,get_digit(a)))
        data.append((img,activate( get_digit(a))))
    return data
            
        


    
     
