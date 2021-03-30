# # import sys

# # for arg in sys.argv: 
# #     print (arg)
# from PIL import Image
# import numpy as np
# print('========== Image p loading ===========')
# img = Image.open('.\data\Tubingen_resized.png')
# p = np.array(img)
# print('Avant reshape : ' + str(p.shape))
# p = np.reshape(p,(512,512,3))
# print('Après reshape p : ' + str(p.shape))

# print('========== Image a loading ===========')
# img = Image.open('.\data\Derschrei_resized.png')
# a = np.array(img)
# print('Avant reshape : ' + str(a.shape))
# a = np.reshape(a,(1,512,512,3))
# print('Après reshape a : ' + str(a.shape))

# print('========== Image x loading ===========')
# iimg = Image.open('.\data\img_white_noise.png')
# x = np.array(img)
# print('shape x : ' + str(x.shape))
# import tensorflow as tf
# from matplotlib import pyplot as plt
# # years = [1950,1960,1970]
# # gdp = [300.2,543.3,1075.9]
# # plt.plot(years, gdp, color='green', linestyle='solid')
# # plt.title("Valeur de gdp")
# # plt.ylabel("Millards de dollars")
# # plt.show()
# content_image = tf.image.decode_jpeg(tf.io.read_file('.\data\Tubingen.jpg'))
# plt.imshow(content_image)
# plt.show()


# import tensorflow as tf
# m0 = tf.random.normal(shape=[2, 3])
# m1 = tf.random.normal(shape=[3, 5])
# e = tf.einsum('ij,jk->ik', m0, m1)
# # output[i,k] = sum_j m0[i,j] * m1[j, k]
# print(e.shape)

# import tensorflow as tf
# import tensorflow_hub as hub
# from functions import *
# import os
# os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
# hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
# hub_module = hub.load(hub_handle)
# content_image = load_img('.\..\data\portrait1.jpg',[224, 224])
# style_image = load_img('.\..\data\picasso.jpg', [224, 224])
# stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]
# tensor_to_image(stylized_image)

import imageio
import numpy as np
from functions import *
from matplotlib import pyplot as plt
import tensorflow as tf
# import tensorflow_datasets as tfds
import keras


# fig = plt.figure(figsize=(14, 10))
img = tf.keras.preprocessing.image.load_img('.\..\data\Magritte.jpg', grayscale=False, color_mode='rgb', target_size=[224,224],
    interpolation='nearest')


plt.subplot(1,1,1)
plt.imshow(img)
plt.show()

