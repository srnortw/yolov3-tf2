# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 21:34:11 2025

@author: Serkan
"""
import zipfile
import tensorflow as tf
from PIL import Image
import cv2 as cv
import numpy as np
import pdb

import matplotlib.pyplot as plt



class get_all_img_samples_c():
    def __init__(self):
        
        self.all_imgs=[]
        
        
    def read_and_get_all(self,resh,resw,sdaq_df,locs_ds):
        
        start=0
        for _,(source,count) in sdaq_df.iterrows():
            zip_file_path = f"tmp/{source}"
            with zipfile.ZipFile(zip_file_path, 'r') as z:
                
                for loc in locs_ds.skip(start).take(count):#(l[start:start+count] for l in all_metad_df_loc):
                    
                    loc=loc.numpy().decode()

                    # import pdb
                    # pdb.set_trace()

                    image_array=tf.image.decode_jpeg(z.read(loc),channels=3)

                    image_array=tf.image.resize(image_array, [resh, resw], method='nearest')

                    image_array = tf.cast(image_array, tf.uint8)
                    # image_data = z.read(loc) #read bytes nostructured data
                    #
                    # # Convert bytes to a numpy array
                    # image_array = np.frombuffer(image_data, np.uint8) # turns into specific position numbers in array
                    # #it contains images metadata ex: size type(jpg,png...) dimensions stuff its special features
                    #
                    # # Decode the image array to OpenCV format
                    # image_array = cv.imdecode(image_array, cv.IMREAD_COLOR)
                    # image_array=cv.cvtColor(image_array,  cv.COLOR_BGR2RGB)
                    # image_array=cv.resize(image_array,(resh,resw))

                    #
                    # plt.imshow(image_array)
                    # plt.show()
                    # cv.imshow('x',image_array)
                    # cv.waitKey(0)
                    # cv.destroyAllWindows()
                    
                    
                    self.all_imgs.append(image_array)
                    
            start+=count
                
        self.all_imgs=np.array(self.all_imgs)
            
        #print(self.all_imgs.shape)
            
        return self.all_imgs

        
        
