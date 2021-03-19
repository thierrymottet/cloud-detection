# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 16:02:38 2021

@author: pc
"""

#Importation des packages nécessaires
from datetime import timedelta
from pathlib import Path
from time import sleep
import numpy as np
import pandas as pd
import plotly_express as px
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
import cv2
import numpy as np
import os
import streamlit as st
import pandas as pd
import seaborn as sns
from multiapp import MultiApp
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input
from keras.utils import np_utils
from PIL import Image
from keras.preprocessing import image
from keras.preprocessing.image import load_img
from tempfile import NamedTemporaryFile
from PIL import Image,ImageEnhance
import matplotlib.pyplot as plt
import re

#Constantes
cheminImages = "./img/"
cheminTextes = "./txt/"
cheminSources = "./sources/"
cheminModeles = "./models/"
cheminVGG16Simple = cheminModeles+"VGG16Simple/c/"
cheminVGG19Simple = cheminModeles+"VGG19Simple/c/"
cheminVGG19Multiple = cheminModeles+"VGG19Multiple/"
cheminRegression = cheminModeles+"Regression/"
cheminImagesTest = "./testimages/"
resumeVGG19Simple = "ResumeModeleVGG19Simple.txt"
resumeVGG19Multi = "ResumeModeleVGG19Multi.txt"
resumeRegression = "ResumeModeleRegression.txt"

# Constants
img_size = 600
width = 600

#Enregistrement en mémoire des fonction déjà chargées
@st.cache(suppress_st_warning=True)

#Fonction utiles à l'application
# %%
@st.cache
def load_data(file):
    bikes_data_path = Path() / file
    data = pd.read_csv(bikes_data_path)
    return data

#--------------------------------------------------FONCTION POUR LE DEEP LEARNING ------------------------------------------------------------------------------
def modify_path(img):
    path = 'train_images_resized/' + str(img)
    return path

def getclass(row):
    if row.Label_Fish == 1:
        return 0
    if row.Label_Flower == 1:
        return 1
    if row.Label_Gravel == 1:
        return 2
    if row.Label_Sugar == 1:
        return 3   

def loadmodel(chemin, fichier):
    # load json and create model
    json_file = open(chemin+fichier+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(chemin+fichier+'.h5')
    st.write("Chargement du modèle")
    # evaluate loaded model on test data
    opt = Adam(lr=0.001)
    model = loaded_model
    return model

#Fonction permettant de charger une image en mémoire
def retournerImage():
    st.set_option('deprecation.showfileUploaderEncoding', False)

    st.info("Il est nécessaire de charger une image pour réaliser une prédiction")
    st.info("Vous obtiendrez le type de forme ainsi que son emplacement sur l'image")

    buffer = st.file_uploader("Charger ici")
    temp_file = NamedTemporaryFile(delete=False)
    if buffer:
        temp_file.write(buffer.getvalue())
        st.image(load_img(temp_file.name), width = width)
        nomfichier=buffer.name
        #Recupération de l'image
        var = load_img(temp_file.name)
    else:
        var=""
        nomfichier=""
    return var,nomfichier
#-----------------------------------------------------------------------------------------------------------------------------------------------------------

#Fonction de la page principal
def welcome():
    st.title("Présentation de l'application")
    st.subheader("Description : \n")
    file10 = open(cheminTextes+"ProjectContext.txt","r+",  encoding="utf-8")
    for line in file10:
            st.write(line)
    st.subheader("Image d'exemple : \n")
    st.image(cheminImages+'Teaser_AnimationwLabels.gif',use_column_width=True)
    st.write("https://www.kaggle.com/c/understanding_cloud_organization")
    st.subheader("Equipe du projet : Promotion BootCamp Janvier 2021 \n")
    st.info("Gregory BEAUME, Thierry Mottet, Thibault REMY") 

#Fonction permettant de selectionner un modele de Deep Learning
def choixModeleML(image):
    im=image[0]
    nomimage=image[1]
    ########################################################################################
    options = ["1. Modèle VGG19 Simple",
    "2. Modèle VGG19 Multiple",
    "3. Modèle de régression (VGG19)"]
    st.subheader("Choix du modèle")
    choixutilisateur = st.selectbox(label = "", options = options)
    ########################################################################################
    if(choixutilisateur==options[0]):
        Modele_TransfertLearning_VGG19Simple(im)
        ResumeModele(resumeVGG19Simple)
    elif(choixutilisateur==options[1]):
        Modele_TransfertLearning_VGG19Multiple(im)
        ResumeModele(resumeVGG19Multi)
    elif(choixutilisateur==options[2]):
        Modele_Regression(im, nomimage)
        ResumeModele(resumeRegression)

#Fonction permettant d'afficher le resume du modèle utilisé pour la prédiction
def ResumeModele(fichier):
    st.subheader("Description du modèle : \n")
    file10 = open(cheminTextes+fichier,"r+",  encoding="utf-8")
    for line in file10:
            st.write(line)

#Fonctions pour la partie exploration des données
def Exploration_formes():
    
    df_count=cheminSources+"df_count.csv"
    df_corr=cheminSources+"df_corr.csv"

    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.title("Exploration\n")

    st.subheader("Répartition des différentes formes\n")
    st.image(cheminImages+'stat/nbImages.jpg',use_column_width=True)
    st.write("Ci-dessous la distribution du nombre de formes par image\n")

    filea=df_count
    df_count = load_data(filea)
    sns.countplot(x ='nb_formes', data = df_count)
    st.pyplot()
    st.write('nombre moyen de formes par images:' , df_count.nb_formes.mean())
    st.write('nombre d images contenant les 4 formes: ', df_count[df_count.nb_formes == 4].count()[0])

    st.subheader("Catégorisation exacte des labels de formes\n")
    st.write("Dataframe présentant le nombre de forme par image et les couples présent sur une image : ")

    file=df_corr
    df_corr = load_data(file)
    st.write(df_corr)

    st.write("Répartition exacte des images par ensemble de formes : ")

    st.set_option('deprecation.showPyplotGlobalUse', False)
    sns.countplot(y = df_corr['Multilabel'], order = df_corr['Multilabel'].value_counts().index)
    st.pyplot()

def Exploration_couleurs():
    st.title("Exploration\n")

    st.subheader("Couleur des images\n")

    st.image(cheminImages+'stat/couleur01.jpg',use_column_width=True)
    st.image(cheminImages+'stat/couleur02.jpg',use_column_width=True)

    file7 = open(cheminTextes+"CouleursImage02.txt","r+",  encoding="utf-8")
    for line in file7:
        st.write(line)

    st.write("\n")

def Exploration_dimensions():
    
    df_train=cheminSources+"df_train.csv"

    st.title("Exploration\n")

    st.subheader("Dimension des images\n")
    st.image(cheminImages+'stat/dimensionImage.jpg',use_column_width=True)
    st.write("\n")

    st.subheader("Dimensions des formes\n")
    st.write("Diagramme -> Tailles moyennes des formes (donc lorsqu'il y a une forme) indépendamment de la forme")

    file2=df_train
    df_train = load_data(file2)
    sns.histplot(df_train[df_train['nb_pixels'] != 0].nb_pixels, bins=20, kde=True,stat = "density")
    st.pyplot()
    nb_form = df_train[df_train['nb_pixels'] != 0].shape[0]
    st.write('Nombre total de formes présentent sur le dataset entrainement:', nb_form)
    st.write('Nombre de pixels moyen par forme:',df_train[df_train['nb_pixels'] != 0].nb_pixels.mean() )

    file5 = open(cheminTextes+"StatistiqueDimensionsFormes.txt","r+",  encoding="utf-8")
    for line in file5:
        st.write(line)

    st.write("Histogramme de fréquence normalisée (densité) d'apparition pour chaque forme : ")

    df_form = df_train[(df_train['nb_pixels'] != 0)]
    sns.set_context(font_scale=2)  
    sns.displot(data=df_form, bins = 20, x='nb_pixels', col="Label",stat = "density")
    st.pyplot()
 
    st.write("Fréquence totale de chaque Label : ")

    df_form['Label'].value_counts(normalize = True)*100

    st.write("Distribution du nombre de pixels selon son Label sous forme de boîte à moustaches")

    sns.boxplot(y='nb_pixels', x='Label', 
                 data=df_form, 
                 width=0.5,
                 palette="colorblind")

    st.pyplot()

    st.write("Comparaison des tailles moyennes des différentes formes")
    df2=df_form.groupby('Label').mean()
    st.write(df2)

    file8 = open(cheminTextes+"DimensionFormes.txt","r+",  encoding="utf-8")
    for line in file8:
        st.write(line)

#Fonction permettant de faire une prediction sur une image (Avec Transformateur)
def Modele_TransfertLearning_VGG19Simple(image):

    model11 = loadmodel(cheminVGG19Simple,'ModeleTRY-ClassSimple_VGG19FitGenerator_Size600')

    if(image != ""):
        img=image
        image_size = img_size

        img = img.resize((image_size,image_size))
        img = np.array(img)
        img = img / 255.0
        img = img.reshape(1,image_size,image_size,3)
        img_class=model11.predict(img)[0] 
        pred_class = list(img_class).index(max(img_class))

        if (pred_class == 0) : 
            pred_class="Fish"
        elif (pred_class == 1) : 
            pred_class="Flower" 
        elif (pred_class == 2) : 
            pred_class="Sugar" 
        elif (pred_class == 3) : 
            pred_class="Gravel"

        st.write("L'image sélectionnée est de la classe : ",pred_class)

#Fonction permettant de faire une prediction sur une image (Avec Transformateur)
def Modele_TransfertLearning_VGG19Multiple(image):

    model11 = loadmodel(cheminVGG19Multiple,'model_multi_greg')

    if(image != ""):
        img=image
        image_size = img_size - 100
        img = img.resize((image_size,image_size))
        img = np.array(img)
        img = img / 255.0
        img = img.reshape(1,image_size,image_size,3)
        img_class=model11.predict(img)[0] 
        pred_class = list(img_class).index(max(img_class))

        if (pred_class == 0) : 
            pred_class="Fish"
        elif (pred_class == 1) : 
            pred_class="Flower" 
        elif (pred_class == 2) : 
            pred_class="Sugar" 
        elif (pred_class == 3) : 
            pred_class="Gravel"

        st.write("L'image sélectionnée est de la classe : ",pred_class)

#Fonction permettant d'afficher la boudingbox
def show_bounding_box(im, bbox, normalised=True, color='r'):
    
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    # Signification de bbox
    img = np.array(im)
    #img = cv2.imread(image)
    img_r = cv2.resize(img, (300,300))

    im=img_r

    x, y, w, h = bbox
    # Convertir les cordonées (x,y,w,h) en (x1,x2,y1,y2)
    x1=x-w/2
    x2=x+w/2
    y1=y-h/2
    y2=y+h/2
    
    # redimensionner en cas de normalisation
    if normalised:
        x1=x1*im.shape[1]
        x2=x2*im.shape[1]
        y1=y1*im.shape[0]
        y2=y2*im.shape[0]

    #On affiche la prédiction, et la valeur réelle de la bouding box
    st.write("BOUDINGBOX prédite :", bbox)

    # Afficher l'image avec la bouding box    
    fig, ax = plt.subplots()
    im = ax.imshow(im)
    x = [x1,x2,x2,x1,x1]
    y = [y1,y1,y2,y2,y1]
    line, = ax.plot(x, y, "b", label = 'Predicted box')
    legend = ax.legend()
    st.pyplot()

#Fonction permettant de renvoyer le score de l'IOU
def list_corners_IOU(y_true, y_pred) : #y_true = list(xmin, xmax, ymin, ymax)
    xmin = max(y_true[0], y_pred[0])
    ymin = max(y_true[2], y_pred[2])
    xmax = min(y_true[1], y_pred[1])
    ymax = min(y_true[3], y_pred[3])
    # son aire :
    interArea = max(0,xmax-xmin+1)*max(0,ymax-ymin+1)
    # les aires des bbox d'entrée :
    trueArea = (y_true[1]-y_true[0]+1)*(y_true[3]-y_true[2]+1)
    predArea = (y_pred[1]-y_pred[0]+1)*(y_pred[3]-y_pred[2]+1)
    
    return (interArea / (trueArea + predArea - interArea))

#Fonction qui affiche deux bouding box, la réelle et la prédite d'une image
def show_img2(img,y_true, bbox):

    img = np.array(img)
    #img_r = cv2.resize(img, (300,300))
    im=img

    xt0,yt0,wt0,ht0 = y_true

    h_min=xt0
    h_max=yt0
    l_min=wt0
    l_max=ht0

    X = 0.5 * (l_min+l_max)
    Y= 0.5 * (h_min+h_max)
    w= (l_max - l_min)
    h= (h_max-h_min)

    h_factor = 1/1400
    w_factor = 1/2100
    xt0 = w_factor* X
    yt0 = h_factor* Y
    wt0 = w_factor* w
    ht0 = h_factor* h

    #On met à jour la variable y_true avec les variables normalisées
    y_true=xt0,yt0,wt0,ht0

    #On affiche la prédiction, et la valeur réelle de la bouding box
    st.write("BOUDINGBOX prédite :", bbox)
    st.write("BOUDINGBOX réelle :", y_true)

    #[y_true[i] for i in range(len(y_true))]
    xt1= (xt0-wt0/2)*2100
    yt1= (yt0-ht0/2)*1400
    xt2= (xt0+wt0/2)*2100
    yt2= (yt0+ht0/2)*1400
    
    xp0,yp0,wp0,hp0 = bbox
    
    #print(xp0,yp0,wp0,hp0)
    xp1= max((xp0-abs(wp0)/2)*2100,0)
    xp2= min((xp0+abs(wp0)/2)*2100,2100)
    yp1= max((yp0-abs(hp0)/2)*1400,0)
    yp2= min((yp0+abs(hp0)/2)*1400,1400)

    #On calcule la valeur de l'IOU
    IOU = list_corners_IOU(y_true, bbox)
    #st.write("Valeur de l'IOU : ", IOU)

    # Afficher l'image avec la bouding box    
    xt1 = [xt1,xt2,xt2,xt1,xt1]
    yt1 = [yt1,yt1,yt2,yt2,yt1]
    xp1 = [xp1,xp2,xp2,xp1,xp1]
    yp1 = [yp1,yp1,yp2,yp2,yp1]

    fig, ax = plt.subplots()
    im = ax.imshow(im)
    line, = ax.plot(xt1, yt1, "r", label = 'True box')
    line2, = ax.plot(xp1, yp1, "b", label = 'Predicted box')
    legend = ax.legend(title= 'IOU = '+str(round(IOU)))
    st.pyplot()

#Fonction permettant de faire une prediction sur l'emplacement de la forme sur l'image (boudingbox)
def Modele_Regression(image, nomimage):

    file=cheminSources+"train_labels.csv"
    model = loadmodel(cheminRegression,'modelVGG19_YOLOpenalise_bboxfish_v3bis_300')

    if(image != ""):
        img = np.array(image)
        #img = cv2.imread(image)
        img_r = cv2.resize(img, (300,300))
        x,y,w,h = model.predict(np.array([img_r]))[0]
        bbox = x,y,w,h

        y_true=()

        train_labels = load_data(file)
        vare=""
        
        i=0
        for nomfichier in train_labels.filename:
            if (nomfichier == nomimage):
                vare="BON"
                e=i
            i=i+1

        if(vare=="BON"):
            xmin=train_labels.xmin[e]
            ymin=train_labels.ymin[e]
            xmax=train_labels.xmax[e]
            ymax=train_labels.ymax[e]
            y_true=(xmin,ymin,xmax,ymax)
            st.write("Nom de l'image : ",nomimage)
            show_img2(image,y_true, bbox)
        else:
            st.write("Nom de l'image : ",nomimage)
            show_bounding_box(image, bbox)

#Fonction permettant de prédire une image chargée
def cloudDetection():
    st.title("Cloud detection")
    st.text('Construit avec Streamlit,VGG19 and OpenCV')
    st.info("Les formes possible sont : Fish, Flower, Sugar, Gravel")

    our_image=retournerImage()

    if(our_image == ""):
        st.write("NOT IMAGE")
    else:
        #On lance une prédiction sur l'image modifiée        
        choixModeleML(our_image)

#Pages de l'application
def foo():
    welcome()

def main():

    menu = ['Introduction', 'Exploration', 'Cloud Detection']
    choice = st.sidebar.selectbox('Menu',menu)

    if choice == 'Cloud Detection':
        cloudDetection()

    elif choice == 'Introduction':
        foo()

    elif choice == 'Exploration':
        app = MultiApp()
        app.add_app("Répartition des différentes formes", Exploration_formes)
        app.add_app("Répartion des couleurs", Exploration_couleurs)
        app.add_app("Répartion des dimensions des formes", Exploration_dimensions)
        app.run()

if __name__ == '__main__':
    main()