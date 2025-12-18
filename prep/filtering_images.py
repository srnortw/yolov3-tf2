# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 20:14:31 2025

@author: Serkan
"""

import pdb
# import tensorflow as tf
import numpy as np
import cv2 as cv 
import matplotlib.pyplot as plt
import math

import tensorflow as tf


class filtering_images_c():
    
    def __init__(self):
        pass
    
    # Example function for augmenting images
    
    #@staticmethod
    #@tf.numpy_function(Tout=[tf.uint8,tf.uint8])
    def ImagePreprocessor(self,image,histogram_moment,m):
        
        
        #org=image
        
        # if image.shape[0]<256:
        #     return image#,org
        
        # histogram_moment=tf.reshape(histogram_moment,[histogram_moment.shape[0],-1])

        #pdb.set_trace()
        #image=cv.cvtColor(image, cv.COLOR_RGB2BGR)
        
        # cv.imshow('iheq',image)
        # cv.waitKey(0)
        
        
 
        
        
        
        
        #print(histogram_moment[3:6])
        
        
        # if histogram_moment[3]<0:
        #     r = cv.equalizeHist(r)
        # if histogram_moment[4]<0:
        #     g=cv.equalizeHist(g)
        # if histogram_moment[5]<0:
        #     b=cv.equalizeHist(b)
        

        if np.mean(histogram_moment[1])<-1:#-1
            
            image = cv.cvtColor(image, cv.COLOR_RGB2YUV) # you can try also yuv (lumminance,blue chroma,red chroma) or hsv
            
            y, u, v = cv.split(image)   
            
            
            clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))#8,8 256
            
            y = clahe.apply(y)
            
            #y = cv.equalizeHist(y)

            image = cv.merge((y,u,v))
            
            image=cv.cvtColor(image, cv.COLOR_YUV2RGB)
            
        
            

       
        #pdb.set_trace()
            
        gamma=1/(1+math.exp(image.shape[0]*np.mean(histogram_moment[0])/m))#sigmoid function
        #gamma=np.mean(histogram_moment[0])/m
        #gamma=gamma*2.19
        #gamma/=0.7
        
        #gamma*=2#(m*(10**-4))
            
        image=np.clip(pow(image/255,gamma)*255,0,255).astype(np.uint8)
        
        
        
        
        # # Convert to grayscale
        # gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        
        # # Apply Canny edge detection
        # edges = cv.Canny(gray, threshold1=100, threshold2=200)
        
        # # Convert edges to 3-channel to match original image dimensions
        # edges_colored = cv.cvtColor(edges, cv.COLOR_GRAY2RGB)
        
        # # Blend edges with the original image
        # alpha = 0.9  # Weight for the original image
        # beta = 0.1   # Weight for the edges
        # image = cv.addWeighted(image, alpha, edges_colored, beta, 0)



        
        # y = cv.Canny(y, threshold1=100, threshold2=200)
        

        #pdb.set_trace()
        #         # Convert to grayscale
        # image_g = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        
        # # Apply Canny edge detection
        # image_g_edges = cv.Canny(image_g, threshold1=100, threshold2=200)
        
        # # Convert edges to 3-channel to match original image dimensions
        # image_edges = cv.cvtColor(image_g_edges, cv.COLOR_GRAY2RGB)
        
        # # Blend edges with the original image
        # alpha = 0.7  # Weight for the original image
        # beta = 0.3   # Weight for the edges
        # image = cv.addWeighted(image, alpha, image_edges, beta, 0)
        
        
        
        if image.shape[0]>64:

            image= cv.medianBlur(image,3)

            image = cv.bilateralFilter(image,25,1.5,1.5)
        
            # gamma=math.pow(100,np.mean(histogram_moment[3:6]))
    
            # #gamma=0.75


        return  image#,org
    

        
    # # Define the augmented pipeline
    # def augmented_pipeline(self,images,imgs_histograms_moments, batch_size=32):
        
        
    #     #pdb.set_trace() 
        
    #     #inputs={"i": images, "hms": imgs_histograms_moments}
        
    #     dataset = tf.data.Dataset.from_tensor_slices((images,imgs_histograms_moments))#
        
    #     # for i in dataset:
    #     #     plt.imshow(i)
    #     #     print(i[0])
    #     #     plt.show()
    #     #     break
        
    #     # def augment_fn(image):
    #     #     #Use tf.numpy_function to apply the augmentation
    #     #     augmented_image = tf.numpy_function(
    #     #         func=self._augment_image,
    #     #         inp=[image],
    #     #         Tout=tf.uint8
    #     #     )
    #     #     augmented_image=self._augment_image(image)
    #     #     # Set the shape of the augmented image (TensorFlow cannot infer it)
    #     #     augmented_image.set_shape(image.shape)
    #     #     return augmented_image        
     
       

    #     #pdb.set_trace()
    #     dataset= dataset.map(filtering_images_c._augment_image)#,num_parallel_calls=tf.data.AUTOTUNE
        
    #     for i in dataset:
    #         print(i)
    #         break
        
    #     x=np.array(list(dataset.as_numpy_iterator()))
            
            
    #     #x=list(dataset)
        
    #     dataset = dataset.batch(batch_size)

    #     # for i in dataset:
    #     #     z=i[0]
    #     #     plt.imshow(z)
    #     #     print(i[0])
    #     #     plt.show()
            
    #     dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)  # Prefetch for efficiency
        
        

    #     pdb.set_trace()  
        
    #     return dataset
    
    
    
    
    
    
    
    
    
    
    
    
    
    

 