
Guide d’installation et utilisation du code

SPEC
Operating System
Windows 10 Famille
GPU
 GeForce GTX 1650 Ti
CUDA Version
10.1

Step 1 : Installation CUDA
https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html
lib : cuDNN = 7.6.4
Step 2：Installation Anaconda
https://docs.anaconda.com/anaconda/install/
Step 3：Installation environnement virtuel
Jouer les commandes suivantes en séquence dans Anaconda Prompt :
conda create -n tf-gpu tensorflow-gpu = 2.1.0
conda activate tf-gpu
conda install tensorflow-gpu-estimator = 2.1.0
conda install python = 3.7.7
conda install pillow = 7.2.0
conda install numpy = 1.17.0
conda install keras-applications = 1.0.8
conda install keras-base = 2.3.1
conda install keras-gpu = 2.3.1
conda install keras-preprocessing = 1.1.0

Step 4 : Répertoire et manipulation

le répertoire data contient les images en entrée et les images générée (+graph)
le répertoire v1 contient le programme
le répertoire graph contient le graph du modèle 
le répertoire input contient les images style et content
Le répertoire output contient les images transfert en .jpg et .gif
le fichier main.py est le programme principale
le fichier functions.py contient les fonctions utilisées par le programmes principales
Ne pas prendre en compte le fichier test.py 

Comment lancer le programme ? 
> Ajouter vos images content et style dans le dossier input 
> Modifier le répertoire dans le fichier main.py
> ouvrir un console de commande
> se placer dans le répertoire du code 
> entrer python main.py 



