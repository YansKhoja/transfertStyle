from PIL import Image
# https://ascendances.wordpress.com/2016/08/03/redimensionner-des-images-avec-python-resize-image/
from resizeimage import resizeimage 
import numpy as np
import keras
import tensorflow as tf
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
    taille = (512,512)
    formatPicture = '.png'
    a = resized(chemin, name, taille, formatPicture, saved, shown)
    
    # Content representation - resize 
    print('======= Tubingen =======')
    chemin = '.\data'
    name = '\Tubingen'
    taille = (512,512)
    formatPicture = '.png'
    p = resized(chemin, name, taille, formatPicture, saved, shown)
    
    # initialise the image synthesis
    print('======= Image generated =======')
    x = generative(chemin, saved, shown)
    
else : 
    model = VGG19(include_top=True, weights='imagenet')
    model.layers.pop()
    model.layers.pop()
    model.layers.pop()
    model.layers.pop()
    model = Model(model.input,model.layers[-1].output)
    keras.utils.plot_model(model,"model_used.png", show_shapes=True)
    
    # learning rate
    lr = 0.1
    
    # Load images
    print('========== Image p loading ===========')
    img = Image.open('.\data\Tubingen_resized.png')
    p = np.array(img)
    p = np.reshape(p,(1,512,512,4))
    print('shape p : ' + str(p.shape))
    
    print('========== Image a loading ===========')
    img = Image.open('.\data\Derschrei_resized.png')
    a = np.array(img)
    print('shape a : ' + str(a.shape))
    
    print('========== Image x loading ===========')
    iimg = Image.open('.\data\img_white_noise.png')
    x = np.array(img)
    print('shape x : ' + str(x.shape))
    
    model_content = model
    model_content.compile(loss='binary_crossentropy', optimizer=SGD(lr, momentum=0.9), metrics=['binary_accuracy'])
    model_content.fit(p, batch_size=None, epochs=1, verbose=1)
    print('model content fitted on image p')
    
    model_style = model
    model_style.compile(loss='binary_crossentropy', optimizer=SGD(lr, momentum=0.9), metrics=['binary_accuracy'])
    # used train_on_batch
    model_style.fit(a, batch_size=None, epochs=1, verbose=1)
    print('model style fitted on image a')
    
    model_generative = model
    model_generative.compile(loss='binary_crossentropy', optimizer=SGD(lr, momentum=0.9), metrics=['binary_accuracy'])
    model_generative.fit(x, batch_size=None, epochs=1, verbose=1)
    print('model generative fitted on image x')
    print('end !')
    
    # get_layer
    
    # Calculer tout les G pour chaque l
    # Calculer tout les A pour chaque l
    # Calculer dE pour chaque l
    # Calculer dlossStyle = si Fij > 0 => somme des produit wl * dE 
    
    
    # Calculer dlossContent = si Fij > O => Fij - Pij 
    
    # dlossTotal = alpha * dlossContent + beta * dlossStyle
    # x = x - lambda * dlossTotal 
    
    
    
    
