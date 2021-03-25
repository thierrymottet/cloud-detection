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
cheminVGG19Simple = cheminModeles+"VGG19Simple/c/"
cheminVGG19Multiple = cheminModeles+"VGG19Multiple/"
cheminRegressionFish = cheminModeles+"Regression/Fish/"
cheminRegressionFlower = cheminModeles+"Regression/Flower/"
cheminRegressionSugar = cheminModeles+"Regression/Sugar/"
cheminRegressionGravel = cheminModeles+"Regression/Gravel/"
cheminYolo = cheminModeles+"YOLO/"
cheminImagesTest = "./testimages/"

#Texte txt (affichage de texte sur les pages)
resumeVGG19Simple = "ResumeModeleVGG19Simple.txt"
resumeVGG19Multi = "ResumeModeleVGG19Multi.txt"
resumeRegression = "ResumeModeleRegression.txt"
resumeYOLO = "ResumeModeleYOLO.txt"
projetcontexte = "ProjectContext.txt"

#CSV (récupéré des différents NoteBooks/Projet)
dfcount = "df_count.csv"
dfcorr = "df_corr.csv"
dftrain = "df_train.csv"
dftrainwithbbox = "train_with_bbox_finalversion.csv"
dftargetdata = "df_targetdata.csv"

#Modele (fichier JSON + H5)
ClassSimple_VGG19 = "ModeleTRY-ClassSimple_VGG19FitGenerator_Size600"
ClassMulti_VGG19 = "model_multi_greg"
modelREGRESSION_VGG19_Fish = "modelREGRESSION_VGG19_Fish"
modelREGRESSION_VGG19_Flower = "modelREGRESSION_VGG19_Flower"
modelREGRESSION_VGG19_Sugar = "modelREG_VGG19_Sugar"
modelREGRESSION_VGG19_Gravel = "modelREG_VGG19_Gravel"
model_YOLO = "model4_YOLO"

# Constantes
img_size = 600
width = 600

#Constantes pour Yolo
nb_class = 0
output_shape = (5, 5)

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
    file10 = open(cheminTextes+projetcontexte,"r+",  encoding="utf-8")
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
    options = [
    #"1. Modèle VGG19 Simple",
    #"1. Modèle VGG19 Multiple",
    "1. Classification VGG19 + régression",
    "2. Modèle YOLO"]
    st.subheader("Choix du modèle")
    choixutilisateur = st.selectbox(label = "", options = options)
    ########################################################################################
    #if(choixutilisateur==options[0]):
    #    Modele_TransfertLearning_VGG19Simple(im)
    #    ResumeModele(resumeVGG19Simple)
    #if(choixutilisateur==options[0]):
    #    Modele_TransfertLearning_VGG19Multiple(im)
    #    ResumeModele(resumeVGG19Multi)
    if(choixutilisateur==options[0]):
        Modele_Regression(im, nomimage)
        ResumeModele(resumeRegression)
    elif(choixutilisateur==options[1]):
        Modele_YOLO(im, nomimage)
        ResumeModele(resumeYOLO)

#Fonction permettant d'afficher le resume du modèle utilisé pour la prédiction
def ResumeModele(fichier):
    st.subheader("Description du modèle : \n")
    file10 = open(cheminTextes+fichier,"r+",  encoding="utf-8")
    for line in file10:
            st.write(line)

#Fonctions pour la partie exploration des données
def Exploration_formes():
    
    df_count=cheminSources+dfcount
    df_corr=cheminSources+dfcorr
    dft = cheminSources+dftrain

    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.title("Exploration\n")
    st.subheader("Répartition des différentes formes\n")
    st.write('\n')
    
    dft = load_data(dft)
    nb_fish = dft[(dft['Label'] == 'Fish') & (dft['nb_pixels'] != 0)].Label.count()
    st.write('Nombre d images contenant la forme Fish: ', nb_fish)
    nb_flower = dft[(dft['Label'] == 'Flower') & (dft['nb_pixels'] != 0)].Label.count()
    st.write('Nombre d images contenant la forme Flower: ', nb_flower)
    nb_gravel = dft[(dft['Label'] == 'Gravel') & (dft['nb_pixels'] != 0)].Label.count()
    st.write('Nombre d images contenant la forme Gravel: ', nb_gravel)
    nb_sugar = dft[(dft['Label'] == 'Sugar') & (dft['nb_pixels'] != 0)].Label.count()
    st.write('Nombre d images contenant la forme Sugar: ', nb_sugar)
    filea=df_count
    df_count = load_data(filea)
    st.write('Nombre total d images : ', pd.DataFrame(df_count.groupby('ImageId')).shape[0])
    df_count = df_count.rename(columns = {'nb_formes':'Nombre de formes'})
    
    st.subheader("Ci-dessous la distribution du nombre de formes par image\n")
    sns.countplot(x ='Nombre de formes', data = df_count).set_title("Nombre d'images par nombre de formes qu'elles contiennent")
    st.pyplot()
    df_count = df_count.rename(columns = {'Nombre de formes' : 'nb_formes'})
    st.write('Nombre moyen de formes par images:' , round(df_count.nb_formes.mean(),2))
    st.write("Nombre d'images contenant les 4 formes: ", round(df_count[df_count.nb_formes == 4].count()[0],2))

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
    
    df_train=cheminSources+dftrain

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

    model11 = loadmodel(cheminVGG19Simple,ClassSimple_VGG19)

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

    model11 = loadmodel(cheminVGG19Multiple, ClassMulti_VGG19)

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
        return pred_class

#########################################################FONCTION POUR LA REGRESSION#######################################################
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

###########################################################################################################################################

#Fonction permettant de faire une prediction sur l'emplacement de la forme sur l'image (boudingbox)
def Modele_Regression(image, nomimage):

    file=cheminSources+dftrainwithbbox

    if(image != ""):

        img = np.array(image)
        #img = cv2.imread(image)
        img_r = cv2.resize(img, (300,300))
        
        y_true=()
        bbox=()

        #Variable permettant de récupérer la classe de l'image
        classe=""

        train_labels = load_data(file)
        vare=""
        
        i=0
        for nomfichier in train_labels.ImageId:
            if (nomfichier == nomimage):
                vare="BON"
                e=i
            i=i+1

        #Si l'image est connue, on affiche la bouding box réelle et la bouding box prédite
        if(vare=="BON"):
            X=train_labels.X[e]
            Y=train_labels.Y[e]
            w=train_labels.w[e]
            h=train_labels.h[e]
            #classe=train_labels.Label[e]
            #st.write("La forme sur l'image est :",classe)

            #On récupère la classe de l'image
            classe = Modele_TransfertLearning_VGG19Multiple(image)
            
            #On réalise une regression suivant le type de forme pour trouver la boudingbox
            if (classe == "Fish"):
                modelFish = loadmodel(cheminRegressionFish, modelREGRESSION_VGG19_Fish)
                model=modelFish
                x,y,w,h = model.predict(np.array([img_r]))[0]
                bbox = x,y,w,h
            elif (classe == "Flower"):
                modelFlower = loadmodel(cheminRegressionFlower, modelREGRESSION_VGG19_Flower)
                model=modelFlower
                x,y,w,h = model.predict(np.array([img_r]))[0]
                bbox = x,y,w,h
            elif (classe == "Sugar"):
                modelSugar = loadmodel(cheminRegressionSugar, modelREGRESSION_VGG19_Sugar)
                model=modelSugar
                x,y,w,h = model.predict(np.array([img_r]))[0]
                bbox = x,y,w,h
            elif (classe == "Gravel"):
                modelGravel = loadmodel(cheminRegressionGravel, modelREGRESSION_VGG19_Gravel)
                model=modelGravel
                x,y,w,h = model.predict(np.array([img_r]))[0]
                bbox = x,y,w,h

            y_true=(X,Y,w,h)
            st.write("Nom de l'image : ",nomimage)
            show_img2(image,y_true, bbox)

        #Si l'image n'est pas connue, on affiche la bouding box prédite uniquement
        else:
            st.write("Nom de l'image : ",nomimage)

            #On récupère la classe de l'image
            classe = Modele_TransfertLearning_VGG19Multiple(image)

            if (classe == "Fish") :
                modelFish = loadmodel(cheminRegressionFish, modelREGRESSION_VGG19_Fish)
                model=modelFish
                x,y,w,h = model.predict(np.array([img_r]))[0]
                bbox = x,y,w,h
            elif (classe == "Flower") : 
                modelFlower = loadmodel(cheminRegressionFlower, modelREGRESSION_VGG19_Flower)
                model=modelFlower
                x,y,w,h = model.predict(np.array([img_r]))[0]
                bbox = x,y,w,h
            elif (classe == "Sugar") : 
                modelSugar = loadmodel(cheminRegressionSugar, modelREGRESSION_VGG19_Sugar)
                model=modelSugar
                x,y,w,h = model.predict(np.array([img_r]))[0]
                bbox = x,y,w,h
            elif (classe == "Gravel") : 
                modelGravel = loadmodel(cheminRegressionGravel, modelREGRESSION_VGG19_Gravel)
                model=modelGravel
                x,y,w,h = model.predict(np.array([img_r]))[0]
                bbox = x,y,w,h
            
            st.write(bbox)
            st.write("L'image sélectionnée est de la classe XXX : ",classe)
            show_bounding_box(image, bbox)

#########################################################FONCTION POUR LE YOLO#############################################################
def transform_netout(y_pred_raw):
    y_pred_xy = (tf.nn.tanh(y_pred_raw[..., 1:3]))
    y_pred_wh = tf.sigmoid(y_pred_raw[..., 3:5])
    y_pred_conf = tf.sigmoid(y_pred_raw[..., :1])
    y_pred_class = tf.sigmoid(y_pred_raw[..., 5:9])
    return tf.concat([y_pred_conf, y_pred_xy, y_pred_wh,y_pred_class], -1)

def generate_yolo_grid(g):
    c_x = tf.cast(tf.reshape(tf.tile(tf.range(g), [g]), (1, g, g)), 'float32')
    c_y = tf.transpose(c_x, (0,2,1))
    return tf.stack([tf.reshape(c_x, (-1, g*g)), tf.reshape(c_y, (-1, g*g))] , -1)

c_grid = generate_yolo_grid(output_shape[0])

def proccess_xy(y_true_raw):
    y_true_xy = ((y_true_raw[..., 1:3]+1)/2 + c_grid)/output_shape[0]
    y_true_wh = y_true_raw[..., 3:5]
    y_true_conf = y_true_raw[..., :1]
    y_true_class = y_true_raw[..., 5:9]
    return tf.concat([y_true_conf, y_true_xy, y_true_wh,y_true_class], -1) 

def pred_bboxes(y, threshold):
    y_xy = tf.cast(y, tf.float32)
    y_xy = tf.expand_dims(y_xy, axis=0)
    y_xy = proccess_xy(y_xy)[0]
    #return y_xy
    bboxes =  sorted(y_xy.numpy(), key=lambda x: x[0], reverse=True)
    bboxes = np.array(bboxes)
    result = bboxes[bboxes[:,0]>threshold]
    #print("result avant ajust:", len(result))
    # on doit ajuster les valeurs pour assurer la présence de classes différentes
    if len(result)== 0:
        # dans ce cas il n'y a aucune box retenue, on doit en mettre une
        kmax = np.argmax(bboxes[:,0]) 
        result = bboxes[kmax,:].reshape([1,9])

    # pour chaque bbox on met toutes les probas de classe à 0 sauf la plus haute
    for k in range(len(result)):
        imax = np.argmax(result[k,5:])
        pmax = result[k,5+imax]
        result[k,5:]=0
        result[k,5+imax]=pmax
        # ensuite, on met la  l la proba de chaque classe à 1 pour celle qui a la proba max, et 0 pour les autres
        # pour éviter des doublons
    for i in range(4):
            kmax = np.argmax(result[:,5+i]*result[:,0])
            pmax = result[kmax,5+i]
            result[:,5+i]=0
            if pmax > 0:
                result[kmax,5+i]=1
        #enfin on vire les bbox qui ne prédisent plus de classes après notre post processing
    #print("result apres adjust", result)
    #result_final = []
    result_final = result[(result[:,5]+result[:,6]+result[:,7]+result[:,8])>0 ]
    #for k in range(len(result)):
        #print(element)
        #if not (np.max(result[k,5:]) == 0):
        #    result_final.append(result[k,:])
    #print(type(result_final))
    return result_final 

@tf.function
def load_image(filepath, resize=(320,320)):
    im = tf.io.read_file( filepath)
    im = tf.image.decode_png(im, channels=3)
    return tf.image.resize(im, resize)

# methode renvoyant pour une image donnée (argument path) le vecteur prédit par notre modèle
def compute_y_pred(imgpath,model, resize=(320,320)):
    im = tf.io.read_file(imgpath)
    im = tf.image.decode_png(im, channels=3)
    #     im_shape = tf.shape(im)
    im = tf.image.resize(im, resize)
    pred = model(np.array([im], dtype=np.float32))[0]
    pred = transform_netout(pred)
    #bboxes_pred = pred_bboxes(pred)
    #print(bboxes_pred)
    return pred

# Retourner la couleur et le label de l'image
def show_img_from_bboxes(imgpath, bboxes ,resize=(320,320)):
    for i in range(bboxes.shape[0]):       
        if bboxes[i,5] == 1:
            col = 'r'
            lab = 'Fish'
        if bboxes[i,6] == 1:
            col = 'b'
            lab = 'Flower'
        if bboxes[i,7] == 1:
            col = 'g'
            lab = 'Gravel'
        if bboxes[i,8] == 1:
            col = 'y'
            lab = 'Sugar'
    return col, lab

def compare_predictions_multi(df,model):

    '''
    Methode qui reçoit en argument 
    2)un dataframe contenant 
        dans la premiere colonne le path d'une image 
        dans sa deuxième colonne les vecteurs decrivant les bounding boxes exactes (entre 1 et 4 vecteurs à 9 variables)
    3) un modele
    la methode choisit 4affiche les boîtes exactes avec leurs labels, ainsi que les boîtes prédites
    '''
    #bbox = []
    plt.figure(figsize = (20,10))
    for j, index in enumerate(np.random.randint(0, df.shape[0], [4])):        
        #image choisie au hasard
        img_name = df.iloc[index,1]
        # on recupere dans les données chargées les bounding boxes attendues
        bboxform = df.iloc[index,2]
        bboxform = np.matrix(bboxform)
        size = int(bboxform.shape[1]/9)
        bboxform = bboxform.reshape((size,9))

        normalised=True

    ###################################################################################################################################################

        im = tf.io.read_file(img_name)
        im = tf.image.decode_png(im, channels=3)
        #im_shape = tf.shape(im)

        if(size==1):
            bboxform1=bboxform[0]

            x11=bboxform1[0, 1]
            y11=bboxform1[0, 2]
            w11=bboxform1[0, 3]
            h11=bboxform1[0, 4]

            retour1=show_img_from_bboxes(img_name, bboxform1)
            couleur1=retour1[0]
            label1=retour1[1]

            Color=couleur1
            Label=label1

            # Signification de bbox
            x = x11
            y = y11 
            w = w11
            h = h11

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
            
            # Afficher l'image avec la bouding box    
            x = [x1,x2,x2,x1,x1]
            y = [y1,y1,y2,y2,y1]

            #Affichage à l'écran
            fig, ax = plt.subplots()
            im = ax.imshow(im)
            title = 'TRUE - ' + img_name[12:]
            plt.title(title)
            line, = ax.plot(x, y, color = Color, Label = Label)
            legend = ax.legend(title= 'Label')
            st.pyplot()

        elif(size==2):
            bboxform1=bboxform[0]
            bboxform2=bboxform[1]

            #VALEUR N°1
            x11=bboxform1[0, 1]
            y11=bboxform1[0, 2]
            w11=bboxform1[0, 3]
            h11=bboxform1[0, 4]

            retour1=show_img_from_bboxes(img_name, bboxform1)
            couleur1=retour1[0]
            label1=retour1[1]

            #VALEUR N°2
            x22=bboxform2[0, 1]
            y22=bboxform2[0, 2]
            w22=bboxform2[0, 3]
            h22=bboxform2[0, 4]

            retour2=show_img_from_bboxes(img_name, bboxform2)
            couleur2=retour2[0]
            label2=retour2[1]

            #Recuperation des couleurs
            Color1=couleur1
            Label1=label1
            Color2=couleur2
            Label2=label2

            #*******************************sous partie 1*************************************
            # Signification de bbox
            x = x11
            y = y11 
            w = w11
            h = h11

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
            
            # Afficher l'image avec la bouding box    
            xa = [x1,x2,x2,x1,x1]
            ya = [y1,y1,y2,y2,y1]

            #*******************************sous partie 2*************************************
            # Signification de bbox
            xb = x22
            yb = y22 
            wb = w22
            hb = h22

            # Convertir les cordonées (x,y,w,h) en (x1,x2,y1,y2)
            x1b=xb-wb/2
            x2b=xb+wb/2
            y1b=yb-hb/2
            y2b=yb+hb/2
            
            # redimensionner en cas de normalisation
            if normalised:
                x1b=x1b*im.shape[1]
                x2b=x2b*im.shape[1]
                y1b=y1b*im.shape[0]
                y2b=y2b*im.shape[0]
            
            # Afficher l'image avec la bouding box    
            xb = [x1b,x2b,x2b,x1b,x1b]
            yb = [y1b,y1b,y2b,y2b,y1b]

            #Affichage à l'écran
            fig, ax = plt.subplots()
            im = ax.imshow(im)
            title = 'TRUE - ' + img_name[12:]
            plt.title(title)
            line, = ax.plot(xa, ya, color = Color1, Label = Label1)
            line2, = ax.plot(xb, yb, color = Color2, Label = Label2)
            legend = ax.legend(title= 'Label')
            st.pyplot()

        elif(size==3):
            bboxform1=bboxform[0]
            bboxform2=bboxform[1]
            bboxform3=bboxform[2]

            #VALEUR N°1
            x11=bboxform1[0, 1]
            y11=bboxform1[0, 2]
            w11=bboxform1[0, 3]
            h11=bboxform1[0, 4]

            retour1=show_img_from_bboxes(img_name, bboxform1)
            couleur1=retour1[0]
            label1=retour1[1]

            #VALEUR N°2
            x22=bboxform2[0, 1]
            y22=bboxform2[0, 2]
            w22=bboxform2[0, 3]
            h22=bboxform2[0, 4]

            retour2=show_img_from_bboxes(img_name, bboxform2)
            couleur2=retour2[0]
            label2=retour2[1]

            #VALEUR N°3
            x33=bboxform3[0, 1]
            y33=bboxform3[0, 2]
            w33=bboxform3[0, 3]
            h33=bboxform3[0, 4]

            retour3=show_img_from_bboxes(img_name, bboxform3)
            couleur3=retour3[0]
            label3=retour3[1]

            #Recuperation des couleurs
            Color1=couleur1
            Label1=label1
            Color2=couleur2
            Label2=label2
            Color3=couleur3
            Label3=label3

            #*******************************sous partie 1*************************************
            # Signification de bbox
            x = x11
            y = y11 
            w = w11
            h = h11

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
            
            # Afficher l'image avec la bouding box    
            xa = [x1,x2,x2,x1,x1]
            ya = [y1,y1,y2,y2,y1]

            #*******************************sous partie 2*************************************
            # Signification de bbox
            xb = x22
            yb = y22 
            wb = w22
            hb = h22

            # Convertir les cordonées (x,y,w,h) en (x1,x2,y1,y2)
            x1b=xb-wb/2
            x2b=xb+wb/2
            y1b=yb-hb/2
            y2b=yb+hb/2
            
            # redimensionner en cas de normalisation
            if normalised:
                x1b=x1b*im.shape[1]
                x2b=x2b*im.shape[1]
                y1b=y1b*im.shape[0]
                y2b=y2b*im.shape[0]
            
            # Afficher l'image avec la bouding box    
            xb = [x1b,x2b,x2b,x1b,x1b]
            yb = [y1b,y1b,y2b,y2b,y1b]


            #*******************************sous partie 3*************************************
            # Signification de bbox
            xc = x33
            yc = y33 
            wc = w33
            hc = h33

            # Convertir les cordonées (x,y,w,h) en (x1,x2,y1,y2)
            x1c=xc-wc/2
            x2c=xc+wc/2
            y1c=yc-hc/2
            y2c=yc+hc/2
            
            # redimensionner en cas de normalisation
            if normalised:
                x1c=x1c*im.shape[1]
                x2c=x2c*im.shape[1]
                y1c=y1c*im.shape[0]
                y2c=y2c*im.shape[0]
            
            # Afficher l'image avec la bouding box    
            xc = [x1c,x2c,x2c,x1c,x1c]
            yc = [y1c,y1c,y2c,y2c,y1c]

            #Affichage à l'écran
            fig, ax = plt.subplots()
            im = ax.imshow(im)
            title = 'TRUE - ' + img_name[12:]
            plt.title(title)
            line, = ax.plot(xa, ya, color = Color1, Label = Label1)
            line2, = ax.plot(xb, yb, color = Color2, Label = Label2)
            line3, = ax.plot(xc, yc, color = Color3, Label = Label3)
            legend = ax.legend(title= 'Label')
            st.pyplot()

        elif(size==4):
            bboxform1=bboxform[0]
            bboxform2=bboxform[1]
            bboxform3=bboxform[2]
            bboxform4=bboxform[3]

            #VALEUR N°1
            x11=bboxform1[0, 1]
            y11=bboxform1[0, 2]
            w11=bboxform1[0, 3]
            h11=bboxform1[0, 4]

            retour1=show_img_from_bboxes(img_name, bboxform1)
            couleur1=retour1[0]
            label1=retour1[1]

            #VALEUR N°2
            x22=bboxform2[0, 1]
            y22=bboxform2[0, 2]
            w22=bboxform2[0, 3]
            h22=bboxform2[0, 4]

            retour2=show_img_from_bboxes(img_name, bboxform2)
            couleur2=retour2[0]
            label2=retour2[1]

            #VALEUR N°3
            x33=bboxform3[0, 1]
            y33=bboxform3[0, 2]
            w33=bboxform3[0, 3]
            h33=bboxform3[0, 4]

            retour3=show_img_from_bboxes(img_name, bboxform3)
            couleur3=retour3[0]
            label3=retour3[1]

            #VALEUR N°4
            x44=bboxform4[0, 1]
            y44=bboxform4[0, 2]
            w44=bboxform4[0, 3]
            h44=bboxform4[0, 4]

            retour4=show_img_from_bboxes(img_name, bboxform4)
            couleur4=retour4[0]
            label4=retour4[1]

            #Recuperation des couleurs
            Color1=couleur1
            Label1=label1
            Color2=couleur2
            Label2=label2
            Color3=couleur3
            Label3=label3
            Color4=couleur4
            Label4=label4

            #*******************************sous partie 1*************************************
            # Signification de bbox
            x = x11
            y = y11 
            w = w11
            h = h11

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
            
            # Afficher l'image avec la bouding box    
            xa = [x1,x2,x2,x1,x1]
            ya = [y1,y1,y2,y2,y1]

            #*******************************sous partie 2*************************************
            # Signification de bbox
            xb = x22
            yb = y22 
            wb = w22
            hb = h22

            # Convertir les cordonées (x,y,w,h) en (x1,x2,y1,y2)
            x1b=xb-wb/2
            x2b=xb+wb/2
            y1b=yb-hb/2
            y2b=yb+hb/2
            
            # redimensionner en cas de normalisation
            if normalised:
                x1b=x1b*im.shape[1]
                x2b=x2b*im.shape[1]
                y1b=y1b*im.shape[0]
                y2b=y2b*im.shape[0]
            
            # Afficher l'image avec la bouding box    
            xb = [x1b,x2b,x2b,x1b,x1b]
            yb = [y1b,y1b,y2b,y2b,y1b]


            #*******************************sous partie 3*************************************
            # Signification de bbox
            xc = x33
            yc = y33 
            wc = w33
            hc = h33

            # Convertir les cordonées (x,y,w,h) en (x1,x2,y1,y2)
            x1c=xc-wc/2
            x2c=xc+wc/2
            y1c=yc-hc/2
            y2c=yc+hc/2
            
            # redimensionner en cas de normalisation
            if normalised:
                x1c=x1c*im.shape[1]
                x2c=x2c*im.shape[1]
                y1c=y1c*im.shape[0]
                y2c=y2c*im.shape[0]
            
            # Afficher l'image avec la bouding box    
            xc = [x1c,x2c,x2c,x1c,x1c]
            yc = [y1c,y1c,y2c,y2c,y1c]

            #*******************************sous partie 4*************************************
            # Signification de bbox
            xd = x44
            yd = y44 
            wd = w44
            hd = h44

            # Convertir les cordonées (x,y,w,h) en (x1,x2,y1,y2)
            x1d=xd-wd/2
            x2d=xd+wd/2
            y1d=yd-hd/2
            y2d=yd+hd/2
            
            # redimensionner en cas de normalisation
            if normalised:
                x1d=x1d*im.shape[1]
                x2d=x2d*im.shape[1]
                y1d=y1d*im.shape[0]
                y2d=y2d*im.shape[0]
            
            # Afficher l'image avec la bouding box    
            xd = [x1d,x2d,x2d,x1d,x1d]
            yd = [y1d,y1d,y2d,y2d,y1d]

            #Affichage à l'écran
            fig, ax = plt.subplots()
            title = 'TRUE - ' + img_name[12:]
            plt.title(title)
            im = ax.imshow(im)
            line, = ax.plot(xa, ya, color = Color1, Label = Label1)
            line2, = ax.plot(xb, yb, color = Color2, Label = Label2)
            line3, = ax.plot(xc, yc, color = Color3, Label = Label3)
            line4, = ax.plot(xd, yd, color = Color4, Label = Label4)
            legend = ax.legend(title= 'Label')
            st.pyplot()

    ###################################################################################################################################################
    
        y_pred = compute_y_pred(img_name,model)
        bboxpredites = pred_bboxes(y_pred, threshold = 0.3)
        bboxpredites = np.matrix(bboxpredites)
        sizepred = len(bboxpredites)
        normalised=True

        imp = tf.io.read_file(img_name)
        imp = tf.image.decode_png(imp, channels=3)

        if(sizepred==1):
            bboxpredites1=bboxpredites[0]

            x11=bboxpredites1[0, 1]
            y11=bboxpredites1[0, 2]
            w11=bboxpredites1[0, 3]
            h11=bboxpredites1[0, 4]

            retour1=show_img_from_bboxes(img_name, bboxpredites1)
            couleur1=retour1[0]
            label1=retour1[1]

            Color=couleur1
            Label=label1

            # Signification de bbox
            x = x11
            y = y11 
            w = w11
            h = h11

            # Convertir les cordonées (x,y,w,h) en (x1,x2,y1,y2)
            x1=x-w/2
            x2=x+w/2
            y1=y-h/2
            y2=y+h/2
            
            # redimensionner en cas de normalisation
            if normalised:
                x1=x1*imp.shape[1]
                x2=x2*imp.shape[1]
                y1=y1*imp.shape[0]
                y2=y2*imp.shape[0]
            
            # Afficher l'image avec la bouding box    
            xa = [x1,x2,x2,x1,x1]
            ya = [y1,y1,y2,y2,y1]

            #Affichage à l'écran
            fig, ax = plt.subplots()
            im = ax.imshow(imp)
            title = 'PREDITE - ' + img_name[12:]
            plt.title(title)
            line, = ax.plot(xa, ya, color = Color, Label = Label)
            legend = ax.legend(title= 'Label')
            st.pyplot()

        elif(sizepred==2):
            bboxpredites1=bboxpredites[0]
            bboxpredites2=bboxpredites[1]

            x11=bboxpredites1[0, 1]
            y11=bboxpredites1[0, 2]
            w11=bboxpredites1[0, 3]
            h11=bboxpredites1[0, 4]
            
            x22=bboxpredites2[0, 1]
            y22=bboxpredites2[0, 2]
            w22=bboxpredites2[0, 3]
            h22=bboxpredites2[0, 4]

            retour1=show_img_from_bboxes(img_name, bboxpredites1)
            couleur1=retour1[0]
            label1=retour1[1]
            Color=couleur1
            Label=label1

            retour2=show_img_from_bboxes(img_name, bboxpredites2)
            couleur2=retour2[0]
            label2=retour2[1]
            Color2=couleur2
            Label2=label2

            # Signification de bbox
            x = x11
            y = y11 
            w = w11
            h = h11

            # Convertir les cordonées (x,y,w,h) en (x1,x2,y1,y2)
            x1=x-w/2
            x2=x+w/2
            y1=y-h/2
            y2=y+h/2
            
            # redimensionner en cas de normalisation
            if normalised:
                x1=x1*imp.shape[1]
                x2=x2*imp.shape[1]
                y1=y1*imp.shape[0]
                y2=y2*imp.shape[0]
            
            # Afficher l'image avec la bouding box    
            xa = [x1,x2,x2,x1,x1]
            ya = [y1,y1,y2,y2,y1]

            # Signification de bbox
            xb = x22
            yb = y22 
            wb = w22
            hb = h22

            # Convertir les cordonées (x,y,w,h) en (x1,x2,y1,y2)
            xb=xb-wb/2
            xb=xb+wb/2
            yb=yb-hb/2
            yb=yb+hb/2
            
            # redimensionner en cas de normalisation
            if normalised:
                x1b=xb*imp.shape[1]
                x2b=xb*imp.shape[1]
                y1b=yb*imp.shape[0]
                y2b=yb*imp.shape[0]
            
            # Afficher l'image avec la bouding box    
            xb = [x1b,x2b,x2b,x1b,x1b]
            yb = [y1b,y1b,y2b,y2b,y1b]

            #Affichage à l'écran
            fig, ax = plt.subplots()
            im = ax.imshow(imp)
            title = 'PREDITE - ' + img_name[12:]
            plt.title(title)
            line, = ax.plot(xa, ya, color = Color, Label = Label)
            line2, = ax.plot(xb, yb, color = Color2, Label = Label2)
            legend = ax.legend(title= 'Label')
            st.pyplot()

        elif(sizepred==3):
            bboxpredites1=bboxpredites[0]
            bboxpredites2=bboxpredites[1]
            bboxpredites3=bboxpredites[2]

            x11=bboxpredites1[0, 1]
            y11=bboxpredites1[0, 2]
            w11=bboxpredites1[0, 3]
            h11=bboxpredites1[0, 4]
            
            x22=bboxpredites2[0, 1]
            y22=bboxpredites2[0, 2]
            w22=bboxpredites2[0, 3]
            h22=bboxpredites2[0, 4]

            x33=bboxpredites3[0, 1]
            y33=bboxpredites3[0, 2]
            w33=bboxpredites3[0, 3]
            h33=bboxpredites3[0, 4]

            retour1=show_img_from_bboxes(img_name, bboxpredites1)
            couleur1=retour1[0]
            label1=retour1[1]
            Color=couleur1
            Label=label1

            retour2=show_img_from_bboxes(img_name, bboxpredites2)
            couleur2=retour2[0]
            label2=retour2[1]
            Color2=couleur2
            Label2=label2

            retour3=show_img_from_bboxes(img_name, bboxpredites3)
            couleur3=retour3[0]
            label3=retour3[1]
            Color3=couleur3
            Label3=label3

            ##########Partie a#############################################################
            # Signification de bbox
            x = x11
            y = y11 
            w = w11
            h = h11

            # Convertir les cordonées (x,y,w,h) en (x1,x2,y1,y2)
            x1=x-w/2
            x2=x+w/2
            y1=y-h/2
            y2=y+h/2
            
            # redimensionner en cas de normalisation
            if normalised:
                x1=x1*imp.shape[1]
                x2=x2*imp.shape[1]
                y1=y1*imp.shape[0]
                y2=y2*imp.shape[0]
            
            # Afficher l'image avec la bouding box    
            xa = [x1,x2,x2,x1,x1]
            ya = [y1,y1,y2,y2,y1]

            ##########Partie b#############################################################
            # Signification de bbox
            xb = x22
            yb = y22 
            wb = w22
            hb = h22

            # Convertir les cordonées (x,y,w,h) en (x1,x2,y1,y2)
            xb=xb-wb/2
            xb=xb+wb/2
            yb=yb-hb/2
            yb=yb+hb/2
            
            # redimensionner en cas de normalisation
            if normalised:
                x1b=xb*imp.shape[1]
                x2b=xb*imp.shape[1]
                y1b=yb*imp.shape[0]
                y2b=yb*imp.shape[0]
            
            # Afficher l'image avec la bouding box    
            xb = [x1b,x2b,x2b,x1b,x1b]
            yb = [y1b,y1b,y2b,y2b,y1b]

            ##########Partie c#############################################################
            # Signification de bbox
            xc = x33
            yc = y33 
            wc = w33
            hc = h33

            # Convertir les cordonées (x,y,w,h) en (x1,x2,y1,y2)
            xc=xc-wc/2
            xc=xc+wc/2
            yc=yc-hc/2
            yc=yc+hc/2
            
            # redimensionner en cas de normalisation
            if normalised:
                x1c=xc*imp.shape[1]
                x2c=xc*imp.shape[1]
                y1c=yc*imp.shape[0]
                y2c=yc*imp.shape[0]
            
            # Afficher l'image avec la bouding box    
            xc = [x1c,x2c,x2c,x1c,x1c]
            yc = [y1c,y1c,y2c,y2c,y1c]


            #Affichage à l'écran
            fig, ax = plt.subplots()
            im = ax.imshow(imp)
            title = 'PREDITE - ' + img_name[12:]
            plt.title(title)
            line, = ax.plot(xa, ya, color = Color, Label = Label)
            line2, = ax.plot(xb, yb, color = Color2, Label = Label2)
            line3, = ax.plot(xc, yc, color = Color3, Label = Label3)
            legend = ax.legend(title= 'Label')
            st.pyplot()

        elif(sizepred==4):
            bboxpredites1=bboxpredites[0]
            bboxpredites2=bboxpredites[1]
            bboxpredites3=bboxpredites[2]
            bboxpredites4=bboxpredites[3]

            x11=bboxpredites1[0, 1]
            y11=bboxpredites1[0, 2]
            w11=bboxpredites1[0, 3]
            h11=bboxpredites1[0, 4]
            
            x22=bboxpredites2[0, 1]
            y22=bboxpredites2[0, 2]
            w22=bboxpredites2[0, 3]
            h22=bboxpredites2[0, 4]

            x33=bboxpredites3[0, 1]
            y33=bboxpredites3[0, 2]
            w33=bboxpredites3[0, 3]
            h33=bboxpredites3[0, 4]

            x44=bboxpredites4[0, 1]
            y44=bboxpredites4[0, 2]
            w44=bboxpredites4[0, 3]
            h44=bboxpredites4[0, 4]

            retour1=show_img_from_bboxes(img_name, bboxpredites1)
            couleur1=retour1[0]
            label1=retour1[1]
            Color=couleur1
            Label=label1

            retour2=show_img_from_bboxes(img_name, bboxpredites2)
            couleur2=retour2[0]
            label2=retour2[1]
            Color2=couleur2
            Label2=label2

            retour3=show_img_from_bboxes(img_name, bboxpredites3)
            couleur3=retour3[0]
            label3=retour3[1]
            Color3=couleur3
            Label3=label3

            retour4=show_img_from_bboxes(img_name, bboxpredites4)
            couleur4=retour4[0]
            label4=retour4[1]
            Color4=couleur4
            Label4=label4

            ##########Partie a#############################################################
            # Signification de bbox
            x = x11
            y = y11 
            w = w11
            h = h11

            # Convertir les cordonées (x,y,w,h) en (x1,x2,y1,y2)
            x1=x-w/2
            x2=x+w/2
            y1=y-h/2
            y2=y+h/2
            
            # redimensionner en cas de normalisation
            if normalised:
                x1=x1*imp.shape[1]
                x2=x2*imp.shape[1]
                y1=y1*imp.shape[0]
                y2=y2*imp.shape[0]
            
            # Afficher l'image avec la bouding box    
            xa = [x1,x2,x2,x1,x1]
            ya = [y1,y1,y2,y2,y1]

            ##########Partie b#############################################################
            # Signification de bbox
            xb = x22
            yb = y22 
            wb = w22
            hb = h22

            # Convertir les cordonées (x,y,w,h) en (x1,x2,y1,y2)
            xb=xb-wb/2
            xb=xb+wb/2
            yb=yb-hb/2
            yb=yb+hb/2
            
            # redimensionner en cas de normalisation
            if normalised:
                x1b=xb*imp.shape[1]
                x2b=xb*imp.shape[1]
                y1b=yb*imp.shape[0]
                y2b=yb*imp.shape[0]
            
            # Afficher l'image avec la bouding box    
            xb = [x1b,x2b,x2b,x1b,x1b]
            yb = [y1b,y1b,y2b,y2b,y1b]

            ##########Partie c#############################################################
            # Signification de bbox
            xc = x33
            yc = y33 
            wc = w33
            hc = h33

            # Convertir les cordonées (x,y,w,h) en (x1,x2,y1,y2)
            xc=xc-wc/2
            xc=xc+wc/2
            yc=yc-hc/2
            yc=yc+hc/2
            
            # redimensionner en cas de normalisation
            if normalised:
                x1c=xc*imp.shape[1]
                x2c=xc*imp.shape[1]
                y1c=yc*imp.shape[0]
                y2c=yc*imp.shape[0]
            
            # Afficher l'image avec la bouding box    
            xc = [x1c,x2c,x2c,x1c,x1c]
            yc = [y1c,y1c,y2c,y2c,y1c]


            ##########Partie d#############################################################
            # Signification de bbox
            xd = x44
            yd = y44 
            wd = w44
            hd = h44

            # Convertir les cordonées (x,y,w,h) en (x1,x2,y1,y2)
            xd=xd-wd/2
            xd=xd+wd/2
            yd=yd-hd/2
            yd=yd+hd/2
            
            # redimensionner en cas de normalisation
            if normalised:
                x1d=xd*imp.shape[1]
                x2d=xd*imp.shape[1]
                y1d=yd*imp.shape[0]
                y2d=yd*imp.shape[0]
            
            # Afficher l'image avec la bouding box    
            xd = [x1d,x2d,x2d,x1d,x1d]
            yd = [y1d,y1d,y2d,y2d,y1d]

            #Affichage à l'écran
            fig, ax = plt.subplots()
            im = ax.imshow(imp)
            title = 'PREDITE - ' + img_name[12:]
            plt.title(title)
            line, = ax.plot(xa, ya, color = Color, Label = Label)
            line2, = ax.plot(xb, yb, color = Color2, Label = Label2)
            line3, = ax.plot(xc, yc, color = Color3, Label = Label3)
            line4, = ax.plot(xd, yd, color = Color4, Label = Label4)
            legend = ax.legend(title= 'Label')
            st.pyplot()

def getdfYOLO():
    file=cheminSources+dftargetdata
    data_df = load_data(file)
    return data_df

############################################################################################################################################

#Fonction permettant de faire une prediction sur l'emplacement de la forme sur l'image + classe de la forme -> YOLO
def Modele_YOLO(image, nomimage):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    df_test=getdfYOLO()
    model = loadmodel(cheminYolo, model_YOLO)
    compare_predictions_multi(df_test,model)

#Fonction principal pour la partie détection d'une forme sur une image
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