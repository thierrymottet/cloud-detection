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

#Constantes
cheminImages = "./img/"
cheminTextes = "./txt/"
cheminSources = "./sources/"
cheminModeles = "./models/"
cheminVGG16Simple = cheminModeles+"VGG16Simple/c/"
cheminVGG19Simple = cheminModeles+"VGG19Simple/c/"
cheminVGG19Multiple = cheminModeles+"VGG19Multiple/"
cheminImagesTest = "./testimages/"

# Constants
image_width = 600
image_height = 600
img_size = 600
batch_size = 64
IMG_SHAPE = 600
batch_size = 32
width = 600

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
    st.write("Loaded model from disk")
    # evaluate loaded model on test data
    opt = Adam(lr=0.001)
    loaded_model.compile(optimizer = opt , loss = 'sparse_categorical_crossentropy' , metrics = ['accuracy'])
    model = loaded_model
    return model

#Fonction permettant de charger une image en mémoire
def retournerImage():
    st.set_option('deprecation.showfileUploaderEncoding', False)

    st.info("It is necessary to load an image in memory in order to be able to make a prediction")

    buffer = st.file_uploader("Image here pl0x")
    temp_file = NamedTemporaryFile(delete=False)
    if buffer:
        temp_file.write(buffer.getvalue())
        st.image(load_img(temp_file.name), width = width)
        var = load_img(temp_file.name)
    return var
#-----------------------------------------------------------------------------------------------------------------------------------------------------------

#Fonction de la page principal
def welcome():        
    st.subheader('Understanding Clouds from Satellite Images\n')
    file10 = open(cheminTextes+"ProjectContext.txt","r+",  encoding="utf-8")
    for line in file10:
            st.write(line)
    st.image(cheminImages+'Teaser_AnimationwLabels.gif',use_column_width=True)
    st.write("https://www.kaggle.com/c/understanding_cloud_organization")

#Fonction permettant de selectionner un modele de Deep Learning
def choixModeleML(image):
    ########################################################################################
    options = ["1. Model VGG16 Simple",
    "2. Model VGG19 Simple",
    "3. Model VGG19 Multiple"]
    st.subheader("Model selection")
    choixutilisateur = st.selectbox(label = "", options = options)
    ########################################################################################
    if(choixutilisateur==options[0]):
        Modele_TransfertLearning_VGG16Simple(image)
    elif(choixutilisateur==options[1]):
        Modele_TransfertLearning_VGG19Simple(image)
    elif(choixutilisateur==options[2]):
        Modele_TransfertLearning_VGG19Multiple(image)

#Fonction qui renvoie sur les différentes page de l'analyse des données (statistiques, visualisation, etc..)
def FirstAnalysis():
    options = ["Première aperçu des données",
    "Première visualisation",
    "Etude Statistique"]
    choixutilisateur = st.selectbox(label = "Que souhaitez-vous savoir ? ", options = options)
    ########################################################################################
    if(choixutilisateur==options[0]):
        FirstAnalysis_PremierApercuDonnees()
    elif(choixutilisateur==options[1]):
        FirstAnalysis_PremierVisualisation()
    elif(choixutilisateur==options[2]):
       FirstAnalysis_EtudeStatistique()

#Fonction renvoyant sur la categorie "Segmentation des données/Premier aperçu des données"
def FirstAnalysis_PremierApercuDonnees():

    file1 = open(cheminTextes+"monfichier.txt","r+",  encoding="utf-8")
    file2 = open(cheminTextes+"monfichier2.txt","r+",  encoding="utf-8")
    file3 = open(cheminTextes+"monfichier3.txt","r+",  encoding="utf-8")

    for line in file1:
        st.write(line)
    
    st.write("\n")

    for line in file2:
        st.write(line)

    st.write("")
    st.image(cheminImages+'stat/image1.jpg',use_column_width=True)

    st.write("\n")

    for line in file3:
        st.write(line)

    st.write("")
    st.image(cheminImages+'stat/image2.jpg',use_column_width=True)

    st.write("\n")

#Fonction renvoyant sur la categorie "Segmentation des données/Premiere visualisation"
def FirstAnalysis_PremierVisualisation():
    st.title("I. Première visualisation\n")

    st.subheader("1) Visualisation d'une image\n")
    #img=retournerImage()
    img=image

    img_color= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # nous initialisons un vecteur pour chaque couleur
    col1 = []
    col2 = []
    col3 = []
    # récupération des 3 niveaux de couleur
    col1.append(img[:,:,0])
    col2.append(img[:,:,1])
    col3.append(img[:,:,2])

    st.write(col1)

#Fonction renvoyant sur la categorie "Segmentation des données/Etude des statistiques"
def FirstAnalysis_EtudeStatistique():
    
    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.write("Nous allons à présent mener quelques études statistiques sommaires, d'abord sur les labels de formes, puis sur les dimensions des formes, et enfin sur les images elles-mêmes.\n")

    st.title("I. Etude statistique\n")

    st.subheader("1) Etude statistique des labels de formes\n")
    st.image(cheminImages+'stat/nbImages.jpg',use_column_width=True)
    st.write("Ci-dessous la distribution du nombre de formes par image\n")
    filea=cheminSources+"df_count.csv"
    df_count = load_data(filea)
    sns.countplot(x ='nb_formes', data = df_count)
    st.pyplot()
    st.write('nombre moyen de formes par images:' , df_count.nb_formes.mean())
    st.write('nombre d images contenant les 4 formes: ', df_count[df_count.nb_formes == 4].count()[0])

    st.subheader("2) Catégorisation exacte des labels de formes\n")
    st.write("Dataframe présentant le nombre de forme par image et les couples présent sur une image : ")

    file=cheminSources+"df_corr.csv"
    df_corr = load_data(file)
    st.write(df_corr)

    st.write("Répartition exacte des images par ensemble de formes : ")

    st.set_option('deprecation.showPyplotGlobalUse', False)
    sns.countplot(y = df_corr['Multilabel'], order = df_corr['Multilabel'].value_counts().index)
    st.pyplot()

    st.subheader("3) Matrice d'adjacence\n")
    sns.heatmap(df_corr.corr(), cbar = True, annot=True, square = True,cmap = 'coolwarm', fmt = '.2f')
    st.pyplot()

    st.title("II. Etude statistique des images\n")
    st.subheader("1) Dimension des images\n")
    st.image(cheminImages+'stat/dimensionImage.jpg',use_column_width=True)
    st.write("\n")
    st.subheader("2) Couleur des images\n")

    file6 = open(cheminTextes+"CouleursImage.txt","r+",  encoding="utf-8")
    for line in file6:
        st.write(line)

    st.image(cheminImages+'stat/couleur01.jpg',use_column_width=True)
    st.image(cheminImages+'stat/couleur02.jpg',use_column_width=True)

    file7 = open(cheminTextes+"CouleursImage02.txt","r+",  encoding="utf-8")
    for line in file7:
        st.write(line)

    st.write("\n")

    st.title("III. Etude statistique des formes\n")

    st.subheader("1) Etude statistique des dimensions des formes\n")
    st.write("Diagramme -> Tailles moyennes des formes (donc lorsqu'il y a une forme) indépendamment de la forme")

    file2=cheminSources+"df_train.csv"
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

    st.subheader("2) Etude statistique des couleurs des formes\n")

#Fonction permettant de faire une prediction sur une image (Avec Transformateur)
def Modele_TransfertLearning_VGG19Simple(image):

    model11 = loadmodel(cheminVGG19Simple,'ModeleTRY-ClassSimple_VGG19FitGenerator_Size600')

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

    st.write("The selected image is from the class : ",pred_class)
       
#Fonction permettant de faire une prediction sur une image (Avec Transformateur)
def Modele_TransfertLearning_VGG16Simple(image):

    model11 = loadmodel(cheminVGG16Simple,'ModeleTRY-ClassSimple_VGG16FitGenerator_Size600')

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

    st.write("The selected image is from the class : ",pred_class)

#Fonction permettant de faire une prediction sur une image (Avec Transformateur)
def Modele_TransfertLearning_VGG19Multiple(image):

    model11 = loadmodel(cheminVGG19Multiple,'model_multi_greg')

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

    st.write("The selected image is from the class : ",pred_class)

#Pages de l'application
def foo():    
    st.title("Context presentation")
    welcome()

def bar1():
    st.title("Informations of Data")
    FirstAnalysis()

def main():

    menu = ['Cloud Detection','About']
    choice = st.sidebar.selectbox('Menu',menu)

    if choice == 'Cloud Detection':

        st.title("Cloud detection")
        st.text('Build with Streamlit,VGG19 and OpenCV')
        st.info("Detectable form possible : Fish, Flower, Sugar, Gravel")

        st.subheader('Original Image')
        our_image=retournerImage()

        enhance_type = st.sidebar.radio('Enhance Type', ['Original', 'Gray-Scale', 'Contrast', 'Brightness', 'Blurring'])

        if enhance_type == 'Gray-Scale':
            new_img = np.array(our_image.convert('RGB'))
            img = cv2.cvtColor(new_img, 1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            our_image=Image.fromarray(gray)
            #Reconversion en RGB (image toujours en Gris quand même) -> permet fonctionnement algo. deep
            our_image = our_image.convert("RGB")
            st.subheader('Modified Image')
            st.image(our_image, width = width)

        if enhance_type == 'Contrast':
            c_rate = st.sidebar.slider('Contrast', 0.5, 3.5)
            enhancer = ImageEnhance.Contrast(our_image)
            img_output = enhancer.enhance(c_rate)
            our_image=img_output
            st.subheader('Modified Image')
            st.image(our_image, width = width)

        if enhance_type == 'Brightness':
            c_rate = st.sidebar.slider('Brightness', 0.5, 3.5)
            enhancer = ImageEnhance.Brightness(our_image)
            img_output = enhancer.enhance(c_rate)
            our_image=img_output
            st.subheader('Modified Image')
            st.image(our_image, width = width)

        if enhance_type == 'Blurring':
            new_img = np.array(our_image.convert('RGB'))
            blur_rate = st.sidebar.slider('Blurring', 0.5, 3.5)
            img = cv2.cvtColor(new_img, 1)
            blur_img = cv2.GaussianBlur(img, (11, 11), blur_rate)
            our_image=Image.fromarray(blur_img)
            st.subheader('Modified Image')
            st.image(our_image, width = width)
        else:
            pass

        #On lance une prédiction sur l'image modifiée        
        choixModeleML(our_image)


    elif choice == 'About':
        st.subheader('Project informations')
        st.info("Nous avons developpé un algorithme de DeepLearning permettant d'identifier une forme de nuage sur une image")
        st.info("Cela réponds à un challenge initié sur la plateforme Kaggle et présenté ci-dessous")
        
        app = MultiApp()
        app.add_app("Context presentation", foo)
        app.add_app("Informations of data", bar1)
        app.run()

if __name__ == '__main__':
    main()