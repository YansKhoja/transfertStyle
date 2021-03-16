#
# Copyright Yans Khoja
#
from PIL import Image
# https://ascendances.wordpress.com/2016/08/03/redimensionner-des-images-avec-python-resize-image/
from resizeimage import resizeimage
from IPython.display import clear_output
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess =  tf.compat.v1.Session(config=config)
tf.executing_eagerly()
# tf.enable_eager_execution()
print("Eager execution: {}".format(tf.executing_eagerly()))

import keras
from keras import *
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.optimizers import SGD
from functions import *
import sys
import time


# Content layer for the feature maps
content_layers = ['block5_conv2']

# Style layer for the feature maps.
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1'
                ]

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

num_layers = (num_content_layers,num_style_layers)

#Chargement des images
content_image = load_img('.\..\data\Turtle.jpg',[224, 224])
style_image = load_img('.\..\data\Kanagawa.jpg', [224, 224])

# content_image = tf.keras.applications.vgg19.preprocess_input(content_image)
# style_image = tf.keras.applications.vgg19.preprocess_input(style_image)

#Les images doivent etre contenu dans un array (1,224,224,3) et en float 32
content_image = np.reshape( content_image,
                            ( 1,np.size(content_image,0),
                                np.size(content_image,1),
                                np.size(content_image,2)))
style_image = np.reshape( style_image,
                          (1, np.size(style_image,0),
                              np.size(style_image,1),
                              np.size(style_image,2)))

content_image = tf.image.convert_image_dtype(content_image, tf.float32)
style_image = tf.image.convert_image_dtype(style_image, tf.float32)

# Affichage des images
# plt.subplot(1, 2, 1)
# imshow(plt, content_image, 'Content Image')
# plt.subplot(1, 2, 2)
# imshow(plt, style_image, 'Style Image')


#Creation du modele
custom_model = create_model(style_layers, content_layers)

# Stockage des feature representation pour chacune des images
style_outputs_target = dict()
content_outputs_target = dict()
style_outputs_target  = get_outputs( style_layers,content_layers,
                                     custom_model, tf.constant(style_image),
                                     num_layers)
content_outputs_target = get_outputs( style_layers,content_layers,
                                      custom_model, tf.constant(content_image),
                                      num_layers)

# Stockage dans un unique dictionnaire
outputs_target = dict()
outputs_target['style_outputs'] = style_outputs_target
outputs_target['content_outputs'] = content_outputs_target

# for name, output in sorted(outputs_target.items()):
#   print("  ", name)
#   print(output)

# Set initial image (Nous partons de l'image content)
init_image = tf.Variable(content_image)

# Usage d'un adam optimizer
opt = tf.optimizers.Adam(learning_rate=0.2, beta_1=0.99, epsilon=1e-1)

# Debut du traitement
content_weight = 1e4
style_weight= 1e-2

loss_weights = (style_weight, content_weight)

start = time.time()

epochs = 10
steps_per_epoch = 100
fig = plt.figure()

step = 0
for n in range(epochs):
  for m in range(steps_per_epoch):
    step += 1
    train_step( opt, init_image, style_layers, content_layers,
                custom_model, num_layers, outputs_target, loss_weights)
    # tensor_to_image(init_image).show()
    print(".")

  print("Train step: {}".format(step))
   # tensor_to_image(init_image).show()
  imshow(plt, init_image)
  plt.draw()
  plt.pause(0.2)
  fig.clear()

end = time.time()
print("Total time: {:.1f}".format(end-start))

imshow(plt, init_image)
plt.show()

