#
# Copyright Yans Khoja
#
from PIL import Image
# https://ascendances.wordpress.com/2016/08/03/redimensionner-des-images-avec-python-resize-image/
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
from matplotlib import pyplot as plt
from tensorflow.python.keras.preprocessing import image as kp_image

# Méthode générant une image en bruit de blanc
def generative(chemin, saved, shown):
    # notes : voir si la generation du bruit est correcte

    # Generer le bruit gaussien de moyenne nulle et d'ecart-type 7 (variance 49)
    noise = np.random.randint(0,255,[224,224,3])
    new_img = Image.new('RGB', (224,224))

    # Charger l'image sous forme d'une matrice de pixels
    img = np.array(new_img)

    # Creer l'image bruitee et l'afficher
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

# Convertir un tenseur en image
def tensor_to_image(tensor):
  tensor = tensor * 255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return Image.fromarray(tensor)

#  Méthode permettant l'affichage de l'image stockée dans un tensor de dimension 4
def imshow(plt, image, title=None):
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)
  plt.imshow(image)
  if title:
     plt.title(title)

# Méthode permettant le chargement et resize de l'image
def load_img(path_to_img, new_shape):
  img = tf.image.decode_image(tf.io.read_file(path_to_img))
  img = tf.image.resize( img, new_shape, method='nearest',
                         preserve_aspect_ratio=False, antialias=True )
  return img

# Méthode permettant de créer le model convolutif basé sur un VGG19 et entrainé sur la base imagenet
# avec en sortie les features maps définis en argument de la méthode
def create_model(style_layers, content_layers):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    outputs = style_outputs + content_outputs
    model = tf.keras.Model([vgg.input],outputs)
    tf.keras.utils.plot_model(model,"model_created.png", show_shapes=True)
    return model

# Méthode permettant de récupèrer les features représentations du modèle pour l'image 
def get_feature_representations(model, image, nbr_layers):
    # calculer les features de l'image
    content_outputs = model(image)
    # Récuperer les feature representations content et style à partir de notre modèle à partir nos layers intermédiaires spécifiques
        # style_features -> Nous voulons récupérer les features representation des layers 
        # 'block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1' pour 
        # l'image style
        # style_features -> Nous voulons récupérer les features representation des layers 
        # 'block5_conv2' pour l'image content
    style_features = [style_layer[0] for style_layer in content_outputs[:nbr_layers]]
    content_features = [content_layer[0] for content_layer in content_outputs[nbr_layers:]]
    return style_features, content_features

# Méthode calculant la matrix de gram
def gram_matrix(input_tensor):
    # print(tf.shape(input_tensor))
    # gram = tf.linalg.einsum('.ik,.kj->lij', input_tensor, input_tensor)
    # input_shape = tf.shape(input_tensor)
    # num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    # return gram/(num_locations)

    # Make the image channels
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)

#  Méthode permettant de récupèrer les features representations final de l'image
def get_outputs(style_layers, content_layers, custom_model, image, num_layers):

    num_content_layers , num_style_layers = num_layers
    
    style_features, content_features = \
        get_feature_representations( custom_model,
                                     image,
                                     num_style_layers)

    # Convertion liste en tenseur
    style_features = \
        [tf.convert_to_tensor(style_feature) for style_feature in style_features]
    content_features = \
        [tf.convert_to_tensor(content_feature) for content_feature in content_features]

    # Calculer la matrice de GRAM pour chacune des features representation des couches styles
    gram_style_features = \
        [gram_matrix(style_feature) for style_feature in style_features]

    # Stokage dans un unique dictionnaire
    content_dict = {content_name:value
                    for content_name, value
                    in zip(content_layers, content_features)}

    style_dict = {style_name:value
                  for style_name, value
                  in zip(style_layers, gram_style_features)}

    outputs = {'content':content_dict, 'style':style_dict}

    return outputs


def style_content_loss(outputs,target,loss_weights, num_layers):

    # Recuperer les  outputs de l'image d'initialisation
    style_outputs = dict(outputs['style'].items())
    content_outputs = dict(outputs['content'].items())


    style_targets = dict(target['style_outputs'].items())
    style_targets_outputs = dict(style_targets['style'].items())
    content_targets = dict(target['content_outputs'].items())
    content_targets_outputs = dict(content_targets['content'].items())


    style_weight, content_weight = loss_weights
    num_style_layers, num_content_layers = num_layers

    # todo : Simplify this single instruction to multiple intruction (too long, possible error root cause)
    style_loss = 0
    for name in style_outputs.keys():
        sqrt = tf.square(style_outputs[name]-style_targets_outputs[name])
        sca = tf.reduce_sum(sqrt)
    style_loss += sca
    # style_loss = tf.add_n([tf.reduce_mean(tf.square(style_outputs[name]-style_targets_outputs[name])) for name in style_outputs.keys()])
    
    style_loss *= (4. * (3 ** 2) * ((224*224) ** 2))
    
    style_loss *= style_weight / float(num_style_layers)

    # todo : Simplify too
    content_loss = 0
    for name in content_outputs.keys():
        sqrt = tf.square(content_outputs[name]-content_targets_outputs[name])
        sca = tf.reduce_sum(sqrt)
    content_loss += sca
    # content_loss = tf.add_n([tf.reduce_mean(tf.square((content_outputs[name]-content_targets_outputs[name]))) for name in content_outputs.keys()])
    
    content_loss *= 1./2.
    # content_loss *= content_weight / float(num_content_layers)

    loss = style_loss + content_loss

    return loss

def clip_0_1(image):
  return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

# Calculer les gradients par rapport à l'image d'entrée
# @tf.function()
def train( opt, image_init, style_layers,
                content_layers, custom_model,
                num_layers,target,loss_weights, total_variation_weight):

   with tf.GradientTape() as tape:
       outputs = get_outputs( style_layers,
                              content_layers,
                              custom_model,
                              image_init,
                              num_layers)
       loss = style_content_loss(outputs,target,loss_weights, num_layers)
       loss += total_variation_weight*tf.image.total_variation(image_init)

   grad = tape.gradient(loss, image_init)

   opt.apply_gradients([(grad, image_init)])
   image_init.assign(clip_0_1(image_init))
   # image_init.assign(tf.clip_by_value(image_init, min_vals, max_vals))

   return image_init

def high_pass_x_y(image):
  x_var = image[:, :, 1:, :] - image[:, :, :-1, :]
  y_var = image[:, 1:, :, :] - image[:, :-1, :, :]

  return x_var, y_var
