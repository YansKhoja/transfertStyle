from PIL import Image
# https://ascendances.wordpress.com/2016/08/03/redimensionner-des-images-avec-python-resize-image/
from resizeimage import resizeimage 
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess =  tf.compat.v1.Session(config=config)
import keras
from keras import *
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.optimizers import SGD
from functions import *
import sys

if sys.argv[1]  == 'init' :

    saved = True
    shown = True
    # Style representation - resize 
    print('======= Derschrei =======')
    chemin = '.\data'
    name = '\Derschrei'
    taille = [224,224]
    formatPicture = '.png'
    a = resized(chemin, name, taille, formatPicture, saved, shown)
    
    # Content representation - resize 
    print('======= Tubingen =======')
    chemin = '.\data'
    name = '\Tubingen'
    taille = (224,224)
    formatPicture = '.png'
    p = resized(chemin, name, taille, formatPicture, saved, shown)
    
    # initialise the image synthesis
    print('======= Image generated =======')
    x = generative(chemin, saved, shown)
    
else : 
    # Content layer for the feature maps
    content_layers = ['block4_conv2']

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
    content_image = load_img('.\..\data\Tubingen.jpg',[224, 224])
    style_image = load_img('.\..\data\Derschrei.jpg', [224, 224])
    
    # pré-processing VGG19
    content_image = tf.keras.applications.vgg19.preprocess_input(content_image)
    style_image = tf.keras.applications.vgg19.preprocess_input(style_image)
    
    
    #Les images doivent être contenu dans un array (1,224,224,3) et en float 32
    content_image = np.reshape(content_image,(1,np.size(content_image,0),np.size(content_image,1),np.size(content_image,2)))
    style_image = np.reshape(style_image,(1,np.size(style_image,0),np.size(style_image,1),np.size(style_image,2)))
    
    content_image = tf.image.convert_image_dtype(content_image, tf.float32)
    style_image = tf.image.convert_image_dtype(style_image, tf.float32)
    
    # #Affichage des images
    # plt.subplot(1, 2, 1)
    # imshow(content_image, 'Content Image')
    # plt.subplot(1, 2, 2)
    # imshow(style_image, 'Style Image')
    # plt.show()
    
    #Création du modèle
    custom_model = create_model(style_layers, content_layers)
    
    # Stockage des feature représentation pour chacune des images 
    style_outputs_target = dict()
    content_outputs_target = dict()
    style_outputs_target  = get_outputs(style_layers,content_layers, custom_model, tf.constant(style_image), num_layers)
    content_outputs_target = get_outputs(style_layers,content_layers, custom_model, tf.constant(content_image), num_layers)
    
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
    opt = tf.optimizers.Adam(learning_rate=5, beta_1=0.99, epsilon=1e-1)
    
    # Début du traitement
    # num_iterations = 1000
    content_weight = 1e3
    style_weight= 1e-2
    
    
    loss_weights = (style_weight, content_weight)
    idx = 0
    train_step(opt, init_image, style_layers, content_layers, custom_model, num_layers, outputs_target, loss_weights)
    train_step(opt, init_image, style_layers, content_layers, custom_model, num_layers, outputs_target, loss_weights)
    train_step(opt, init_image, style_layers, content_layers, custom_model, num_layers, outputs_target, loss_weights)
    train_step(opt, init_image, style_layers, content_layers, custom_model, num_layers, outputs_target, loss_weights)
    train_step(opt, init_image, style_layers, content_layers, custom_model, num_layers, outputs_target, loss_weights)
    train_step(opt, init_image, style_layers, content_layers, custom_model, num_layers, outputs_target, loss_weights)
    train_step(opt, init_image, style_layers, content_layers, custom_model, num_layers, outputs_target, loss_weights)
    train_step(opt, init_image, style_layers, content_layers, custom_model, num_layers, outputs_target, loss_weights)
    train_step(opt, init_image, style_layers, content_layers, custom_model, num_layers, outputs_target, loss_weights)
    train_step(opt, init_image, style_layers, content_layers, custom_model, num_layers, outputs_target, loss_weights)
    train_step(opt, init_image, style_layers, content_layers, custom_model, num_layers, outputs_target, loss_weights)
    train_step(opt, init_image, style_layers, content_layers, custom_model, num_layers, outputs_target, loss_weights)
    train_step(opt, init_image, style_layers, content_layers, custom_model, num_layers, outputs_target, loss_weights)
    train_step(opt, init_image, style_layers, content_layers, custom_model, num_layers, outputs_target, loss_weights)
    train_step(opt, init_image, style_layers, content_layers, custom_model, num_layers, outputs_target, loss_weights)
    train_step(opt, init_image, style_layers, content_layers, custom_model, num_layers, outputs_target, loss_weights)
    train_step(opt, init_image, style_layers, content_layers, custom_model, num_layers, outputs_target, loss_weights)
    train_step(opt, init_image, style_layers, content_layers, custom_model, num_layers, outputs_target, loss_weights)
    train_step(opt, init_image, style_layers, content_layers, custom_model, num_layers, outputs_target, loss_weights)
    train_step(opt, init_image, style_layers, content_layers, custom_model, num_layers, outputs_target, loss_weights)
    train_step(opt, init_image, style_layers, content_layers, custom_model, num_layers, outputs_target, loss_weights)
    train_step(opt, init_image, style_layers, content_layers, custom_model, num_layers, outputs_target, loss_weights)
    train_step(opt, init_image, style_layers, content_layers, custom_model, num_layers, outputs_target, loss_weights)
    train_step(opt, init_image, style_layers, content_layers, custom_model, num_layers, outputs_target, loss_weights)
    train_step(opt, init_image, style_layers, content_layers, custom_model, num_layers, outputs_target, loss_weights)
    train_step(opt, init_image, style_layers, content_layers, custom_model, num_layers, outputs_target, loss_weights)
    train_step(opt, init_image, style_layers, content_layers, custom_model, num_layers, outputs_target, loss_weights)
    train_step(opt, init_image, style_layers, content_layers, custom_model, num_layers, outputs_target, loss_weights)
    train_step(opt, init_image, style_layers, content_layers, custom_model, num_layers, outputs_target, loss_weights)
    train_step(opt, init_image, style_layers, content_layers, custom_model, num_layers, outputs_target, loss_weights)
    train_step(opt, init_image, style_layers, content_layers, custom_model, num_layers, outputs_target, loss_weights)
    train_step(opt, init_image, style_layers, content_layers, custom_model, num_layers, outputs_target, loss_weights)
    init_image = train_step(opt, init_image, style_layers, content_layers, custom_model, num_layers, outputs_target, loss_weights)
    print(init_image)
    # print(init_image.shape)
    
    tensor_to_image(init_image).show()
    
    
    #Affichage des images
    # plt.subplot(1, 2, 1)
    # imshow(plt,init_image, 'Image')
    # plt.show()