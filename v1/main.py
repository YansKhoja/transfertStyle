#
# Copyright Yans Khoja
#

#  Bibliothèque d'apprentissage profond & manipulation de tenseur
import tensorflow as tf
import keras
from keras import *
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.optimizers import SGD

#  Bibliothèque classique
import numpy as np
from matplotlib import pyplot as plt

# Recueil de fonctions utilisés dans le programme principale
from functions import *

#  Bibliothèque système
import time
import sys

# Config GPU
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess =  tf.compat.v1.Session(config=config)

# Chargement des images
content_image = tf.keras.preprocessing.image.load_img('.\..\data\Labrador.jpg', color_mode='rgb')
style_image = tf.keras.preprocessing.image.load_img('.\..\data\Kandinsky.jpg', color_mode='rgb')

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
content_layers = ['block4_conv2']

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
opt = tf.optimizers.Adam(learning_rate=0.02)

#  Stockage des poids des loss content et loss style
content_weight = 1e4
style_weight= 1e-2
loss_weights = (style_weight, content_weight)

#  Variable de la perte de variation totale
total_variation_weight = 30

# ========= Debut du traitement ==========
print('========= Debut du traitement ==========')
start = time.time()

epochs = 1
steps_per_epoch = 100
step = 0
for n in range(epochs):
  for m in range(steps_per_epoch):
    step += 1
    train( opt, init_image, style_layers, content_layers,
                custom_model, num_layers, outputs_target, loss_weights, total_variation_weight)
    print(".",end='')
  print("Train step: {}".format(step))

end = time.time()
print("Total time: {:.1f}".format(end-start))
print('========= Fin du traitement ==========')

# Utile pour l'affichage
fig = plt.figure(figsize=(14, 10))

# Analyse perte de variation totale sur les axes verticales et horizontales
x_deltas_content_image, y_deltas_content_image = high_pass_x_y(content_image)
x_deltas_image, y_deltas_image = high_pass_x_y(init_image)

init_image = tf.image.resize(
    init_image, [size_originale_content[0],size_originale_content[1]] , method='nearest', preserve_aspect_ratio=False,
    antialias=False, name=None)

plt.subplot(3,3,1)
imshow(plt, init_image,'Transfert')
plt.subplot(3,3,2)
imshow(plt, style_image,'Style')
plt.subplot(3,3,3)
imshow(plt, content_image,'Content')
plt.subplot(3,3,4)
imshow(plt,clip_0_1(2*y_deltas_content_image+0.5), "Horizontal Deltas: Original")
plt.subplot(3, 3, 5)
imshow(plt,clip_0_1(2*x_deltas_content_image+0.5), "Vertical Deltas: Original")
plt.subplot(3,3,7)
imshow(plt,clip_0_1(2*y_deltas_image+0.5), "Horizontal Deltas: Styled")
plt.subplot(3, 3, 8)
imshow(plt,clip_0_1(2*x_deltas_image+0.5), "Vertical Deltas: Styled")
plt.show()






