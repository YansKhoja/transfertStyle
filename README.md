# TransfertStyle

TransfertStyle a Python program to use GPU tensorflow computation 

### Example de résultats

<img src="https://github.com/user-attachments/assets/9eee9322-59ef-44c0-804d-d4b1fced3e1a" width="512" height="512"/>

# Guide d’installation et utilisation du programme

### Dependances
- Operating System Windows 10 Famille
- GPU GeForce GTX 1650 Ti
- CUDA Version 10.1

##  Step 1 : Installation CUDA

https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html

lib version : cuDNN = 7.6.4

##  Step 2：Installation Anaconda

https://docs.anaconda.com/anaconda/install/

## Step 3：Installation environnement virtuel

Jouer les commandes suivantes en séquence dans Anaconda Prompt :

```bash
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
```

## Step 4 : Répertoires et manipulation

* Le répertoire data contient les images en entrée et les images générée (+graph)
* Le répertoire v1 contient le programme
* Le répertoire graph contient le graph du modèle 
* Le répertoire input contient les images style et content
* Le répertoire output contient les images transfert en .jpg et .gif
* Le fichier main.py est le programme principal
* Le fichier functions.py contient les fonctions utilisées par le programmes principales

Remarque : Ne pas prendre en compte le fichier test.py 

## Comment lancer le programme ?

1) Ajouter vos images content et style dans le dossier input 
2) Modifier le répertoire dans le fichier main.py
3) Ouvrir un console de commande
4) Se placer dans le répertoire du code 
5) Entrer la commande suivante :
    ```bash 
    python main.py
    ```



