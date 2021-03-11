from PIL import Image
# https://ascendances.wordpress.com/2016/08/03/redimensionner-des-images-avec-python-resize-image/
from resizeimage import resizeimage 
import numpy as np
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
    content_image = load_img('.\data\Tubingen.jpg',[224, 224])
    style_image = load_img('.\data\Derschrei.jpg', [224, 224])
    
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
    
    # # Get the style and content feature representations (from our specified intermediate layers)
    # style_features, content_features = get_feature_representations(custom_model, style_image, content_image, num_style_layers)
    
    # # Convertion liste en tenseur
    # style_features = [tf.convert_to_tensor(style_feature) for style_feature in style_features] 
    # content_features = [tf.convert_to_tensor(content_feature) for content_feature in content_features] 
    
    # # Get gram matrix
    # gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]
    
    
    # content_dict = {content_name:value 
    #                 for content_name, value 
    #                 in zip(content_layers, content_features)}

    # style_dict = {style_name:value
    #               for style_name, value
    #               in zip(style_layers, gram_style_features)}
    
    # outputs_target = {'content':content_dict, 'style':style_dict}
    
    outputs_target = get_outputs(style_layers,content_layers, custom_model, style_image, content_image, num_layers)
    
    # print('Target -> Styles:')
    # for name, output in sorted(outputs_target['style'].items()):
    #   print("  ", name)
    #   print("    shape: ", tf.shape(output))
    #   print("    min: ", tf.reduce_min(output))
    #   print("    max: ", tf.reduce_max(output))
    #   print("    mean: ", tf.reduce_mean(output))
    #   print(output)
    
    # print("Target -> Contents:")
    # for name, output in sorted(outputs_target['content'].items()):
    #   print("  ", name)
    #   print("    shape: ", tf.shape(output))
    #   print("    min: ", tf.reduce_min(output))
    #   print("    max: ", tf.reduce_max(output))
    #   print("    mean: ", tf.reduce_mean(output))
    #   print(output)
    
    # Set initial image (Nous partons de l'image content)
    init_image = content_image
    
    # We  use Adam Optimizer
    opt = tf.optimizers.Adam(learning_rate=5, beta_1=0.99, epsilon=1e-1)
    
    # Début du traitement
    # num_iterations = 1000
    content_weight = 1e3
    style_weight= 1e-2
    
    # Create config
    loss_weights = (style_weight, content_weight)
    
    train_step(opt,init_image,style_layers,content_layers, custom_model, style_image, content_image, num_layers,outputs_target,loss_weights)
    
    tensor_to_image(image)