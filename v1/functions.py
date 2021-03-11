# copyright Yans Khoja
from PIL import Image
# https://ascendances.wordpress.com/2016/08/03/redimensionner-des-images-avec-python-resize-image/
from resizeimage import resizeimage 
import numpy as np
import tensorflow as tf
from tensorflow.keras import *
import keras
from keras import *
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.optimizers import SGD
from functions import *
import sys

# function resized returne l'image retaillée dans le format voulu
def resized(chemin, name, taille, formatPicture, saved, shown):
    # notes : /!\ Voir pour étendre l'image sur l'ensemble de la résolution même si déterioration de l'image
    
    #Charger l'image
    print(chemin + name + formatPicture)
    img = Image.open(chemin + name + formatPicture)
    img_resized = img.resize(taille)

    #Afficher l'image chargée redimensionné
    # Récupérer et afficher la taille de l'image (en pixels)
    w, h = img_resized.size
    print("Largeur : {} px, hauteur : {} px".format(w, h))
    # Afficher son mode de quantification
    print("Format des pixels : {}".format(img_resized.mode))
    
    print('Transformation => RGBA -> RGB')
    img_rezised_convert = img_resized.convert('RGB')
    
    
    # Récupérer les valeurs de tous les pixels sous forme d'une matrice
    img_rezised_convert_array = np.array(img_rezised_convert)
    # Afficher la taille de la matrice de pixels
    print("Taille de la matrice de pixels : {}".format(img_rezised_convert_array.shape))
    
    
    if shown == True : 
        img_rezised_convert.show()
    
    if saved == True :
        img_rezised_convert.save(chemin + name + '_resized' + formatPicture, img_rezised_convert.format)
    
    return resized

def generative(chemin, saved, shown):
    # notes : voir si la génération du bruit est correcte
    
    # Générer le bruit gaussien de moyenne nulle et d'écart-type 7 (variance 49)
    noise = np.random.randint(0,255,[224,224,3])
    new_img = Image.new('RGB', (224,224))
    
    # Charger l'image sous forme d'une matrice de pixels
    img = np.array(new_img)

    # Créer l'image bruitée et l'afficher
    generative = img + noise

    noisy_img = Image.frombuffer('RGB',[224,224], generative)
    
    w, h = noisy_img.size
    print("Largeur : {} px, hauteur : {} px".format(w, h))
    print("Format des pixels : {}".format(noisy_img.mode))
    noisy_img_array = np.array(noisy_img)
    print("Taille de la matrice de pixels : {}".format(noisy_img_array.shape))
    
    if saved == True :
        noisy_img.save(chemin + '\img_white_noise.jpg', noisy_img.format)
    
    if shown == True : 
        noisy_img.show()
        
    return generative

def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return Image.fromarray(tensor)

def imshow(plt, image, title=None):
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)
  plt.imshow(image)
  if title:
     plt.title(title)
  
def load_img(path_to_img, new_shape):  
  img = tf.image.decode_image(tf.io.read_file(path_to_img))
  img = tf.image.resize(img, new_shape, method='nearest', preserve_aspect_ratio=False, antialias=False )
  print(img)
  return img

def create_model(style_layers, content_layers):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    print(vgg.input)
    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    outputs = style_outputs + content_outputs
    model = tf.keras.Model([vgg.input],outputs)
    # keras.utils.plot_model(model,"model_created.png", show_shapes=True)
    # print('Modèle créé')
    return model

def get_feature_representations(model, image, nbr_layers):
    
    # compute content and style features
    content_outputs = model(image)
    
    # Get the style and content feature representations from our model
    style_features = [style_layer[0] for style_layer in content_outputs[:nbr_layers]]
    content_features = [content_layer[0] for content_layer in content_outputs[nbr_layers:]]
    # modifier pour récuperer le style_features et content_feature sur la même output donc sur la même image
    #  Utiliser les get_feature_reprensentation dans le get style content loss en le jouant sur leur comme l'autre pour avoir les features targets
    return style_features, content_features

def gram_matrix(input_tensor):
  result = tf.linalg.einsum('ijc,ijd->cd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_locations)


def get_outputs(style_layers, content_layers, custom_model, image, num_layers):
    
    num_content_layers , num_style_layers = num_layers
    # Get the style and content feature representations (from our specified intermediate layers)
    style_features, content_features = get_feature_representations(custom_model, image, num_style_layers)
    
    # Convertion liste en tenseur
    style_features = [tf.convert_to_tensor(style_feature) for style_feature in style_features] 
    content_features = [tf.convert_to_tensor(content_feature) for content_feature in content_features] 
    
    # Get gram matrix
    gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]
            
    content_dict = {content_name:value 
                    for content_name, value 
                    in zip(content_layers, content_features)}

    style_dict = {style_name:value
                  for style_name, value
                  in zip(style_layers, gram_style_features)}
    
    outputs = {'content':content_dict, 'style':style_dict}
    
    # print('Function get output')
    # print('Styles:')
    # for name, output in sorted(outputs['style'].items()):
    #   print("  ", name)
    #   print("    shape: ", tf.shape(output))
    #   print("    min: ", tf.reduce_min(output))
    #   print("    max: ", tf.reduce_max(output))
    #   print("    mean: ", tf.reduce_mean(output))
    #   # print(output)
    
    # print("Contents:")
    # for name, output in sorted(outputs['content'].items()):
    #   print("  ", name)
    #   print("    shape: ", tf.shape(output))
    #   print("    min: ", tf.reduce_min(output))
    #   print("    max: ", tf.reduce_max(output))
    #   print("    mean: ", tf.reduce_mean(output))
    #   # print(output)
    
    return outputs

def style_content_loss(outputs,target,loss_weights, num_layers):
    # print('==============style_content_loss=============')
      
    # Récyperer les  outputs de l'image d'initialisation
    style_outputs = dict(outputs['style'].items())
    content_outputs = dict(outputs['content'].items())
    
    # print('----style_output----')
    # for key, valeur in style_outputs.items():
    #     print(key)
    #     print(valeur)
    # print('')
    
    # print('----content_output----')
    # for key, valeur in content_outputs.items():
    #     print(key)
    #     print(valeur)
    # print('')
    
    style_targets = dict(target['style_outputs'].items())
    style_targets_outputs = dict(style_targets['style'].items())
    content_targets = dict(target['content_outputs'].items())
    content_targets_outputs = dict(content_targets['content'].items())
    
    # print('----style_target----')
    # for key, valeur in style_targets_outputs.items():
    #     print(key)
    #     print(valeur)
    # print('')
    
    # print('----content_target----')
    # for key, valeur in content_targets_outputs.items():
    #     print(key)
    #     print(valeur)
    # print('')
    
    style_weight, content_weight = loss_weights 
    num_style_layers, num_content_layers = num_layers

    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets_outputs[name])**2) for name in style_outputs.keys()])
    # print('style_loss : '+str(style_loss))
    # print('')
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets_outputs[name])**2) for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    # print('content_loss : '+str(content_loss))
    # print('')
    
    loss = style_loss + content_loss
    
    # print('loss : '+str(loss))
    # print('')
    # print('content_targets : '+str(content_targets))
    # print('')
    
    return loss

def clip_0_1(image):
  return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


# Compute gradients according to input image
@tf.function()
def train_step(opt, image_init, style_layers,content_layers, custom_model, num_layers,target,loss_weights):
   # print('=========== train_step =============')
   # print("opt : " + str(opt))
   # print("image : "+str(image_init))
   # print("style_layers : "+str(style_layers))
   # print("content_layers : " + str(content_layers))
   # print("custom_model : "+str(custom_model))
   # print("num_layers : "+str(num_layers))
   # print("target" + str(target))
   # print("loss_weights" + str(loss_weights))
   idx = idx + 1 
   with tf.GradientTape() as tape:
       outputs =  get_outputs(style_layers,content_layers, custom_model, image_init, num_layers)
       loss = style_content_loss(outputs,target,loss_weights, num_layers)
   print('=========== train_step' + str(idx) + ' =============')
   print("outputs : ")
   print(outputs)
   print("loss : ")  
   print(loss)
   grad = tape.gradient(loss, image_init)
   print("grad : ")
   print(grad)
   opt.apply_gradients([(grad, image_init)])
   image_init.assign(clip_0_1(image_init))
   
   return image_init
