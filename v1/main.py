#
# Copyright Yans Khoja
#

#  Bibliothèque d'apprentissage profond & manipulation de tenseur
import tensorflow as tf
import keras
from keras import *
from tensorflow.keras.applications import vgg19
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.optimizers import SGD

#  Bibliothèque classique
import numpy as np
from matplotlib import pyplot as plt

# Image et animation
import PIL.Image

# Recueil de fonctions utilisés dans le programme principale
from functions import *

#  Bibliothèque système
import time
import sys
import os

# Config GPU
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess =  tf.compat.v1.Session(config=config)

# Chargement des images
content_image = tf.keras.preprocessing.image.load_img('.\..\data\input\Turtle.jpg', color_mode='rgb')
style_image = tf.keras.preprocessing.image.load_img('.\..\data\input\Kanagawa.jpg', color_mode='rgb')

# Récupérer la taille des images d'origine
size_originale_content = np.size(content_image,0) , np.size(content_image,1) 
size_originale_style = np.size(style_image,0) , np.size(style_image,1) 

# Convertir les images doivent être contenu dans un array (1,224,224,3)
content_image = np.reshape(content_image,
                            (1,np.size(content_image,0),
                                np.size(content_image,1),
                                np.size(content_image,2)))
style_image = np.reshape(style_image,
                          (1, np.size(style_image,0),
                              np.size(style_image,1),
                              np.size(style_image,2)))

# Re-tailler les images
content_image = tf.image.resize(content_image, [224,224], method='nearest')
style_image = tf.image.resize(style_image, [224,224], method='nearest')

# Conversion en float 32
content_image = tf.image.convert_image_dtype(content_image, tf.float32)
style_image = tf.image.convert_image_dtype(style_image, tf.float32)


# Listes des "feature maps" à récupérer pour l'image content
content_layers = ['block5_conv2']

# Listes des "feature maps" à récupérer pour l'image content
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1'
                ]

# Récupération de la taille des deux listes
num_content_layers = len(content_layers)
num_style_layers = len(style_layers)
num_layers = (num_content_layers,num_style_layers)

# Création du modèle
custom_model = create_model(style_layers, content_layers)

# Stockage des "feature representation" pour chacune des images
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

# Initialisation de l'image à styliser 
init_image = tf.Variable(content_image)

# Usage d'un adam optimizer
opt = tf.optimizers.Adam(learning_rate=0.025)

#  Stockage des poids des loss content et loss style
content_weight = 1e15
style_weight= 1e-5
loss_weights = (style_weight, content_weight)

#  Variable de la perte de variation totale
total_variation_weight = 50

# ========= Debut du traitement ==========
print('========= Debut du traitement ==========')
start = time.time()

epochs = 15
steps_per_epoch = 100
step = 0
fname = "image_generée_iteration_%d.jpg" % 0
tensor_to_image(init_image).save(".\..\data\output\jpg\_" + fname)
for n in range(epochs):
  for m in range(steps_per_epoch):
    step += 1
    train( opt, init_image, style_layers, content_layers,
                custom_model, num_layers, outputs_target, loss_weights, total_variation_weight)
    init_image = init_image
    print(".",end='')
  print("Train step: {}".format(step))
  n = n + 1
  fname = "image_generée_iteration_%d.jpg" % n
  tensor_to_image(init_image).save(".\..\data\output\jpg\_" + fname)

end = time.time()
print("Total time: {:.1f}".format(end-start))
print('========= Fin du traitement ==========')

plt.figure(1,[100,100])
init_image = tf.image.resize(init_image, size_originale_content, method='nearest')
content_image = tf.image.resize(content_image, size_originale_content, method='nearest')
style_image = tf.image.resize(style_image, size_originale_style, method='nearest')
plt.subplot(2,3,5)
imshow(plt,init_image, 'Image final')
plt.subplot(2,3,1)
imshow(plt,content_image, 'Image réaliste')
plt.subplot(2,3,3)
imshow(plt,style_image, 'Image style')
plt.show()

# Création d'une animation
List= []
for n in range(epochs) :
    fname = ".\..\data\output\jpg\_image_generée_iteration_%d.jpg" % n
    img = PIL.Image.open(fname)
    fname = ".\..\data\output\gif\_image_generée_iteration_%d.gif" % n
    img.save(fname)
    print("Image: {}".format(n))
    img = PIL.Image.open(fname)
    List.append(img)
List[0].save('.\..\data\output\gif\style_transfert.gif',save_all=True, append_images=List[1:], optimize=False, duration=200, loop=0)


