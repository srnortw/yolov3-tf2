# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 03:24:23 2025

@author: Serkan
"""
import tensorflow as tf
import cv2 as cv
import numpy as np


import pdb

#from concurrent.futures import ThreadPoolExecutor

        
class images_properties_c():
    def __init__(self,channelq,num_bins):
        self.channelq=channelq
        self.num_bins=num_bins
        
    # Function to compute histogram for an image
    def compute_histogram(self,image):
        
        channels_histograms=[cv.calcHist([image], [i], None, [self.num_bins], [0,self.num_bins]).flatten() for i in range(self.channelq)]
        
        channels_histograms=np.array(channels_histograms) #(channelq,num_bins)
        
        for i in range(self.channelq):
            channels_histograms[i,:]=cv.normalize(channels_histograms[i,:],None, alpha=1, norm_type=cv.NORM_L1).flatten()
            
        channels_histograms=np.moveaxis(channels_histograms,0,-1) #(num_bins,channelq)

        return channels_histograms


    def compute_histograms_moments(self,imgs_histograms):

        # pixel range 0 to 255(8bit)
        pixel_ranges = tf.expand_dims(tf.transpose(tf.range(0.0, self.num_bins)), -1)
        # first_moment

        means = tf.reduce_sum(pixel_ranges * imgs_histograms, axis=0, keepdims=True)

        # second moment
        dss = tf.reduce_sum((pixel_ranges - means) ** 2, axis=0, keepdims=True)
        variances = dss / self.num_bins
        stds = tf.math.sqrt(variances)

        # third moment
        dcs = tf.reduce_sum((pixel_ranges - means) ** 3, axis=0, keepdims=True)
        skewnesses = dcs / ((self.num_bins - 1) * stds ** 3)

        # fourth moment
        dqs = tf.reduce_sum((pixel_ranges - means) ** 4, axis=0, keepdims=True)

        parameter = (self.num_bins * self.num_bins + 1) / (
                    (self.num_bins - 1) * (self.num_bins - 2) * (self.num_bins - 3))

        bias = -3 * ((self.num_bins - 1) ** 2) / ((self.num_bins - 2) * (self.num_bins - 3))

        kurtosises = parameter * (dqs / (stds ** 4)) + bias  # you can add stds**4 epsilon to not go 0

        imgs_histograms_moments = tf.concat([means, stds, skewnesses, kurtosises], axis=0)

        # print(imgs_histograms.shape)
        # #pixel range 0 to 255(8bit)
        # pixel_ranges=np.expand_dims(np.arange(0,self.num_bins).T,-1)
        # #first_moment
        #
        # means=np.sum(pixel_ranges*imgs_histograms,axis=-2,keepdims=True)
        #
        #
        # #second moment
        # dss=np.sum((pixel_ranges-means)**2,axis=-2,keepdims=True)
        # variances=dss/self.num_bins
        # stds=np.sqrt(variances)
        #
        #
        # #third moment
        # dcs=np.sum((pixel_ranges-means)**3,axis=-2,keepdims=True)
        # skewnesses=dcs/((self.num_bins-1)*stds**3)
        #
        #
        # #fourth moment
        # dqs=np.sum((pixel_ranges-means)**4,axis=-2,keepdims=True)
        #
        # parameter=(self.num_bins*self.num_bins+1)/((self.num_bins-1)*(self.num_bins-2)*(self.num_bins-3))
        #
        # bias=-3*((self.num_bins-1)**2)/((self.num_bins-2)*(self.num_bins-3))
        #
        # kurtosises=parameter*(dqs/(stds**4))+bias # you can add stds**4 epsilon to not go 0
        #
        #
        # imgs_histograms_moments=np.concatenate((means,stds,skewnesses,kurtosises),axis=-2).astype(np.float32)
        #

        return imgs_histograms_moments