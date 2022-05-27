import tensorflow as tf
import math

def tiling(latent_features, f_ch, factor):
## Ex) VGG Pool1 : 112 x 112 x 64 --> 224 x 224 x 16, factor : 2 
    batch, height, width, ch = latent_features.shape
    features = latent_features
    features = tf.reshape(features,[batch,height,width,int(ch/f_ch),f_ch])
    features = tf.transpose(features,[0,3,1,2,4])
    features = tf.reshape(features,[batch, factor, factor,height,width,f_ch])
    features = tf.transpose(features,[0,1,3,2,4,5])
    features = tf.reshape(features,[batch, factor*height,factor*width,f_ch])
    return features, factor

def detiling(latent_features, factor, shape):
## Ex) VGG Pool1 : 224 x 224 x 16 --> 112 x 112 x 64, factor : 2 
    if factor==1:
      return latent_features
    batch, h,w,ch = latent_features.shape
    f_height, f_width, f_ch = shape
    features = latent_features
    features = tf.reshape(features,[batch,factor,f_height,factor,f_width,ch])
    features = tf.transpose(features,[0,1,3,2,4,5])
    features = tf.reshape(features,[batch,factor*factor,f_height,f_width,ch])
    features = tf.transpose(features,[0,2,3,1,4])
    features = tf.reshape(features,[batch,f_height,f_width,f_ch])
    return features
