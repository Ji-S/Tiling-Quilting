import tensorflow as tf

def nXn_quilting(latent_features, f_ch, factor):
## Ex) VGG Pool1 : 112 x 112 x 64 --> 224 x 224 x 16, factor : 2 
    batch, height, width, ch = latent_features.shape
    remain = int((height * width * ch) / factor / factor)
    features = tf.reshape(latent_features,shape = [batch, remain, factor, factor])
    features = tf.transpose(features, [0,2,3,1])
    features = tf.reshape(features, shape=[batch, factor, factor, height*width, f_ch])
    features = tf.transpose(features,[0,3,1,2,4])
    features = tf.reshape(features,[batch, height, width, factor, factor, f_ch])
    features = tf.transpose(features,[0,1,3,2,4,5])
    features = tf.reshape(features,[batch, height*factor, width*factor, f_ch])
    return features, [batch, height, width, ch], factor

def nXn_dequilting(latent_features, factor, shape):
## Ex) VGG Pool1 : 224 x 224 x 16 --> 112 x 112 x 64, factor : 2 
    batch, height, width, ch = shape
    f_ch =latent_features.shape[-1]
    features = tf.reshape(latent_features,[batch, height, factor, width, factor,f_ch])
    features = tf.transpose(features,[0,1,3,2,4,5])
    features = tf.reshape(features,[batch, height*width, factor, factor, f_ch])
    features = tf.transpose(features,[0,2,3,1,4])
    features = tf.reshape(features,[batch, factor, factor,-1])
    features = tf.transpose(features,[0,3,1,2])
    features = tf.reshape(features, [batch, height, width, ch])
    return features