from PIL import Image
# https://ascendances.wordpress.com/2016/08/03/redimensionner-des-images-avec-python-resize-image/
from resizeimage import resizeimage 
import numpy as np

# function resized returne l'image retaillée dans le format voulu
def resized(chemin, name, taille, formatPicture, saved, shown):
    # notes : /!\ Voir pour étendre l'image sur l'ensemble de la résolution même si déterioration de l'image
    
    #Charger l'image
    print(chemin + name + formatPicture)
    img = Image.open(chemin + name + formatPicture)
    img_resized = resizeimage.resize_crop(img, taille)

    #Afficher l'image chargée redimensionné
    # Récupérer et afficher la taille de l'image (en pixels)
    w, h = img_resized.size
    print("Largeur : {} px, hauteur : {} px".format(w, h))
    # Afficher son mode de quantification
    print("Format des pixels : {}".format(img.mode))

    # Récupérer les valeurs de tous les pixels sous forme d'une matrice
    resized = np.array(img_resized)
    # Afficher la taille de la matrice de pixels
    print("Taille de la matrice de pixels : {}".format(resized.shape))
    
    resized = Image.frombuffer('RGB',[512,512], np.array(img_resized))
    
    if shown == True : 
        img_resized.show()
    
    if saved == True :
        img_resized.save(chemin + name + '_resized' + formatPicture, img_resized.format)
    
    return resized

def generative(chemin, saved, shown):
    # notes : voir si la génération du bruit est correcte
    
    # Générer le bruit gaussien de moyenne nulle et d'écart-type 7 (variance 49)
    noise = np.random.randint(0,255,[512,512,3])
    new_img = Image.new('RGB', (512,512))
    
    # Charger l'image sous forme d'une matrice de pixels
    img = np.array(new_img)

    # Créer l'image bruitée et l'afficher
    generative = img + noise

    noisy_img = Image.frombuffer('RGB',[512,512], generative)
    
    w, h = noisy_img.size
    print("Largeur : {} px, hauteur : {} px".format(w, h))
    print("Format des pixels : {}".format(noisy_img.mode))
    noisy_img_array = np.array(noisy_img)
    print("Taille de la matrice de pixels : {}".format(noisy_img_array.shape))
    
    if saved == True :
        noisy_img.save(chemin + '\img_white_noise.png', noisy_img.format)
    
    if shown == True : 
        noisy_img.show()
        
    return generative