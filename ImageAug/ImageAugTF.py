import tensorflow as tf
import tensorflow.contrib.eager as tfe
tf.enable_eager_execution()
import random
import requests


def compute_image_tensor_mean_std(image_tensor,axis=[0,1,2,3]):
  batch_mean,batch_variance = tf.nn.moments(image_tensor,axis)
  return(batch_mean.numpy(),batch_variance.numpy())

def normalize_tensor_with_mean_std(input_tensor,mean=0,std=255):
  return tf.divide(tf.subtract(input_tensor, [mean]), [std])

def read_tensor_from_image_url(url,input_height=32,input_width=32):
    image_tensor = tf.image.decode_jpeg(requests.get(url).content, channels=3, name="jpeg_reader")
    image_tensor = tf.cast(image_tensor, tf.float32)
    image_tensor = tf.expand_dims(image_tensor, 0)
    image_tensor = tf.image.resize_bilinear(image_tensor, [input_height, input_width])
    image_tensor = normalize_tensor_with_mean_std(image_tensor)
    return image_tensor

def flip_left_to_right():
  def apply(image_tensor):
    return tf.image.flip_left_right(image_tensor)
  return apply
  
def flip_up_down():
  def apply(image_tensor):
    return tf.image.flip_up_down(image_tensor)
  return apply

def rotate(angle_in_deg=30):
  def apply(image_tensor):
    import math
    deg = angle_in_deg * math.pi/180
    img_rotate = tf.contrib.image.rotate(image_tensor,deg)
    return img_rotate
  return apply

def pad(pad_size = 10,mode="CONSTANT",constant_values=0):
  def apply(image_tensor):
    paddings = tf.constant([[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]])
    return tf.pad(image_tensor, paddings, mode,constant_values=0)
  return apply

def rand_crop(crop_h =100,crop_w =100):
  def apply(image_tensor):
     return tf.image.random_crop(image_tensor,[image_tensor.shape[0],crop_h,crop_w,image_tensor.shape[3]])
  return apply

def random_pad_crop(pad_size = 10,mode="CONSTANT",constant_values=0):
  def apply(image_tensor):
    pad_tensor = pad_image_tensor(pad_size,mode,constant_values)(img_tensor)
    return rand_crop_image_tensor(crop_h =image_tensor.shape[1],crop_w =image_tensor.shape[2])(pad_tensor)
  return apply

def cutout(h_pixel=4,w_pixel=4,v_l=0, v_h=255, patch_gaussian=False):

  def apply(image_tensor):
      img_num = image_tensor.shape[0]
      img_h = image_tensor.shape[1]
      img_w = image_tensor.shape[2]
      img_c = image_tensor.shape[3]

      def cutout_single_image(input_img_orig):
        input_img = tf.identity(input_img_orig)
        input_img = tf.Variable(input_img)
        
        top=0
        left=0
        
        while True:
          left = random.randint(0, img_w)
          top = random.randint(0, img_h)

          if left + w_pixel <= img_w and top + h_pixel <= img_h:
            break

        indices=[]

        for i in range(top, top + h_pixel, 1):
          for j in range(left, left + w_pixel, 1):
            indices.append([i,j]) 
        
        if patch_gaussian:
            updates = tf.random.uniform(shape=[w_pixel*h_pixel,img_c],minval=v_l,maxval=v_h)
        else:
            fill_val = tf.math.reduce_mean(input_img)
            updates = tf.fill([w_pixel*h_pixel,img_c],fill_val)

        tf.scatter_nd_update(input_img,indices ,updates)
        input_img_tensor = tf.convert_to_tensor(input_img)

        return input_img_tensor
      
      aug_tensor = tf.map_fn(cutout_single_image, image_tensor)
      
      return aug_tensor
    
  return apply

class CompositeImageAug(object):
  def __init__(self,augmentations=[(flip_left_to_right,0.5),(rotate(30),0.5)]):
    self.aug = lambda x:x
    if len(augmentations) > 0:
      for au,prob in augmentations:
        self.add_aug(au,prob)

  def compose(self,g, f):
    def h(x):
        return g(f(x))
    return h

  def add_aug(self,oper,prob=0.5):
    def func_prob(img):
      if random.random() < prob:
        return oper(img)
      else:
        return img
    self.aug = self.compose(func_prob,self.aug)
    return self

  def transform(self,img):
    return self.aug(img)



def  flip_left_to_right_image_tensor(img_tensor):
  return flip_left_to_right()(img_tensor)

def  flip_up_down_image_tensor(img_tensor):
  return flip_up_down()(img_tensor)

def pad_image_tensor(img_tensor,pad_size = 10,mode="CONSTANT",constant_values=0):
  return pad(pad_size,mode,constant_values)(img_tensor)
  
def rand_crop_image_tensor(img_tensor,pad_size = 10,mode="CONSTANT",constant_values=0):
  return random_pad_crop(pad_size,mode,constant_values)(img_tensor)

def random_pad_crop_image_tensor(img_tensor,pad_size = 10,mode="CONSTANT",constant_values=0):
  return random_pad_crop(pad_size,mode,constant_values)(img_tensor)

def rotate_image_tensor(img_tensor,angle_in_deg=30):
  return rotate(angle_in_deg)(img_tensor)

def apply_cutout_on_image_tensor(img_tensor,h_pixel=4,w_pixel=4,v_l=0, v_h=255, patch_gaussian=False):
  return cutout(h_pixel=4,w_pixel=4,v_l=0, v_h=255, patch_gaussian=False)(img_tensor)

