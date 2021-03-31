# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 16:02:38 2021

@author: pc
"""

#Importation des packages nécessaires

import numpy as np
import pandas as pd
import plotly_express as px
import streamlit as st
import cv2
import numpy as np
import os
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras.models import model_from_json
from keras.preprocessing.image import load_img
from tempfile import NamedTemporaryFile
import matplotlib.pyplot as plt
from multiapp import MultiApp
from pathlib import Path

#Chemins
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
projetcontexte2 = "ProjectContext2.txt"

#CSV (récupérés des différents NoteBooks)
dfcount = "df_count.csv"
dfcorr = "df_corr.csv"
dftrain = "df_train.csv"
dftrainwithbbox = "train_with_bbox_finalversion.csv"
dftargetdata = "df_targetdata.csv"

#Modèles (fichiers JSON + H5)
ClassMulti_VGG19 = "model2_classi_multi"
modelREGRESSION_VGG19_Fish = "model3_reg_Fish"
modelREGRESSION_VGG19_Flower = "model3_reg_Flower"
modelREGRESSION_VGG19_Sugar = "model3_reg_Sugar"
modelREGRESSION_VGG19_Gravel = "model3_reg_Sugar"
model_YOLO = "model4_YOLO"

#Constantes pour Yolo
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



#--- INTRODUCTION ----------------------------------------------------------------------------------------------------------------------------


def welcome():
    st.title("Présentation de l'application")
    file10 = open(cheminTextes+projetcontexte,"r+",  encoding="utf-8")
    file11 = open(cheminTextes+projetcontexte2,"r+",  encoding="utf-8")
    for line in file10:
            st.write(line)
    st.image(cheminImages+'Teaser_AnimationwLabels.gif',use_column_width=True)
    st.write("https://www.kaggle.com/c/understanding_cloud_organization")
    for line in file11:
            st.write(line)
    st.subheader("Promotion BootCamp DS Janvier 2021 \n")
    st.info("Equipe projet : Gregory BEAUME, Thierry MOTTET, Thibault REMY")



#--- EXPLORATION DONNEES ---------------------------------------------------------------------------------------------------------------------


def Exploration_Labels():
    
    dft = cheminSources+dftrain
    df_count=cheminSources+dfcount
    df_corr=cheminSources+dfcorr

    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.title("Labels\n")
    
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

    st.write("Fréquence de présence de chaque masque : ")
    df_form = dft[(dft['nb_pixels'] != 0)]
    st.write(df_form['Label'].value_counts(normalize = True).apply(lambda x: str(round(100*x,2)) +' %'))

    filea=df_count
    df_count = load_data(filea)
    df_count = df_count.rename(columns = {'nb_formes':'Nombre de formes'})
    st.subheader("Répartition des images selon le nombre exact de masques qu'elles contiennent :")
    sns.countplot(x ='Nombre de formes', data = df_count)
    st.pyplot()
    df_count = df_count.rename(columns = {'Nombre de formes' : 'nb_formes'})
    st.write('Nombre moyen de formes par images:' , round(df_count.nb_formes.mean(),2))
    st.write("Nombre d'images contenant les 4 formes: ", round(df_count[df_count.nb_formes == 4].count()[0],2))
    st.subheader("Répartition des images selon les combinaisons de masques qu'elles contiennent : ")

    # st.set_option('deprecation.showPyplotGlobalUse', False)
    # sns.countplot(x = df_corr['Multilabel'], order = df_corr['Multilabel'].value_counts().index)
    # st.pyplot()
    st.image(cheminImages+'stat/formes4.jpg',width = 600, use_column_width=False)

def Exploration_Images():
    st.title("Images")
    st.write('\n')
    st.write('Nombre : ', 5546)
    st.write('Dimension : ',(1400,2100,3))
    st.write("Scatterplot des niveaux de couleur moyens de chaque image :")
    st.image(cheminImages+'stat/couleur01.jpg',width = 400, use_column_width=False)
    st.info("Pour gagner en temps de traitement, on pourrait presque considérer que nos images sont en niveau de gris.")
    
    st.write("Projection des niveaux de couleur moyens :\n")
    st.write("\n")
    st.image(cheminImages+'stat/couleur02.jpg',use_column_width=True)
    st.info("Mise en évidence de quelques outliers, qui se sont effectivement révélés être des photographies défectueuses")



def Exploration_Taille_Masques():
    
    df_train=cheminSources+dftrain

    st.title("Taille des masques de forme\n")
    st.write('\n')

    st.subheader("Répartition des surfaces (en pixels) des masques encodés :")
    file2=df_train
    df_train = load_data(file2)
    sns.histplot(df_train[df_train['nb_pixels'] != 0].nb_pixels, bins=20, kde=True,stat = "density")
    st.pyplot()
    nb_form = df_train[df_train['nb_pixels'] != 0].shape[0]
 
    st.subheader("Répartition des surfaces (en pixels) des masques encodés suivant leur label : ")
    df_form = df_train[(df_train['nb_pixels'] != 0)]
    sns.set_context(font_scale=2)  
    sns.displot(data=df_form, bins = 20, x='nb_pixels', col="Label",stat = "density")
    st.pyplot()
 
    st.subheader("Boxplot associés :")
    sns.boxplot(y='nb_pixels', x='Label', 
                 data=df_form, 
                 #width=0.5,
                 palette="colorblind")

    st.pyplot()


def Exploration_Couleur_Masques():
    return 0



#--------------------------------------------------------- DETECTION + REGRESSION ----------------------------------------------------------------


# Chargement d'un modèle
def loadmodel(chemin, fichier):
    json_file = open(chemin+fichier+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(chemin+fichier+'.h5')
    return loaded_model

# Fonction permettant de charger une image en mémoire
def retournerImage():
    st.set_option('deprecation.showfileUploaderEncoding', False)

    st.info("Il est nécessaire de charger une image pour réaliser une prédiction")

    buffer = st.file_uploader("Charger ici :")
    temp_file = NamedTemporaryFile(delete=False)
    if buffer:
        temp_file.write(buffer.getvalue())
        st.image(load_img(temp_file.name), width = 600)
        nomfichier=buffer.name
        #Recupération de l'image
        var = load_img(temp_file.name)
    else:
        var=""
        nomfichier=""
    return var,nomfichier

# Sélection du modèle
def choixModeleML(image):
    im=image[0]
    nomimage=image[1]
    options = [
    "1. Classification VGG19 + régression",
    "2. Modèle YOLO"]
    st.subheader("Choix du modèle")
    choixutilisateur = st.selectbox(label = "", options = options)
    if(choixutilisateur==options[0]):
        Modele_Regression(im, nomimage)
        ResumeModele(resumeRegression)
    elif(choixutilisateur==options[1]):
        Modele_YOLO(im, nomimage)
        ResumeModele(resumeYOLO)

# Fonction principale pour la partie détection
def cloudDetection():
    st.title("Détection de formes nuageuses")
    our_image=retournerImage()

    if(our_image == ""):
        st.write("NO IMAGE")
    else:
        #On lance une prédiction sur l'image modifiée        
        choixModeleML(our_image)


###########  DETECTION MULTILABELS + REGRESSION  ###########

# Détection multi-labels
def Modele_TransferLearning_VGG19Multiple(image):

    model11 = loadmodel(cheminVGG19Multiple, ClassMulti_VGG19)

    if(image != ""):
        img=image
        image_size = 500
        img = img.resize((image_size,image_size))
        img = np.array(img)/255.0
        img = img.reshape(1,image_size,image_size,3)
        img_class=model11.predict(img)[0] 
        pred_class = list(img_class)

    return(pred_class)

# Régression des bbox
def Modele_Regression(image, nomimage, ):
    
    if(image != ""):
        pred_class = Modele_TransferLearning_VGG19Multiple(image)
        df=load_data(cheminSources+dftrainwithbbox)
        img2100 = np.array(image)
        img_input = cv2.resize(img2100, (300,300))
        
        dico_models = {'Fish' : (cheminRegressionFish, modelREGRESSION_VGG19_Fish),
                        'Flower' : (cheminRegressionFlower, modelREGRESSION_VGG19_Flower),
                        'Gravel' : (cheminRegressionGravel, modelREGRESSION_VGG19_Gravel),
                        'Sugar' : (cheminRegressionSugar, modelREGRESSION_VGG19_Sugar)}
        
        thresh = 0.5
        bboxes_pred = {}
        for clss, prob in zip(['Fish', 'Flower', 'Gravel', 'Sugar'], pred_class) :
            if prob > thresh :
                model=loadmodel(dico_models[clss][0],dico_models[clss][1])
                bbox_pred = model(np.expand_dims(img_input,0))
                bboxes_pred[clss] = bbox_pred
                
        liste_bbox =[]
        for label in bboxes_pred.keys() :
            bbox=[1]
            for i in range(4) :
                bbox.append(bboxes_pred[label][0,i].numpy())
            for i in ['Fish', 'Flower', 'Gravel', 'Sugar'] :
                if i == label :
                    bbox.append(1)
                else : 
                    bbox.append(0)
            liste_bbox.append(bbox)

        if nomimage in df.ImageId.values :
            df = df[df['ImageId']==nomimage] # df contient les true_bbox de l'image (mais pas que)
            im_in_train_folder = True
            truebboxes = []
            for clss in df.Label :
                array = df[df['Label']==clss][['X','Y','w','h']].values[0]
                truebboxes.append([1,array[0],array[1],array[2],array[3],0,0,0,0])
                if clss == "Fish" :
                    truebboxes[-1][5] = 1
                elif clss == 'Flower' :
                    truebboxes[-1][6] = 1
                elif clss == 'Gravel' :
                    truebboxes[-1][7] = 1
                elif clss == 'Sugar' :
                    truebboxes[-1][8] = 1
            truebboxes = np.matrix(truebboxes)
            show_img_with_bboxes_from_img2(img_input, truebboxes, np.matrix(liste_bbox), resize=(300,300))
        else :
            show_img_with_bboxes_from_img(img_input, np.matrix(liste_bbox), resize=(300,300))

# Résumé du modèle utilisé pour la prédiction
def ResumeModele(fichier):
    st.subheader("Description du modèle : \n")
    st.write('\n')
    file10 = open(cheminTextes+fichier,"r+",  encoding="utf-8")
    for line in file10:
            st.write(line)



#-------------------------------------------------------------   YOLO   -----------------------------------------------------------

# YOLO
def Modele_YOLO(image, nomimage):
    
    if(image != ""):
        st.set_option('deprecation.showPyplotGlobalUse', False)
        model = loadmodel(cheminYolo, model_YOLO)
        img = np.array(image)
        # prédiction des bboxes 
        target = compute_y_pred_from_img(img,model)
        bboxes = pred_bboxes(target, threshold = 0.3)
        bboxes = np.matrix(bboxes)
        #chargement du data_train :
        df=cheminSources+dftrainwithbbox
        df=load_data(df)

        if nomimage in df.ImageId.values :
            truebboxes = []
            df = df[df['ImageId']==nomimage] # df contient les true_bbox de l'image (mais pas que)
            for clss in df.Label :
                array = df[df['Label']==clss][['X','Y','w','h']].values[0]
                truebboxes.append([1,array[0],array[1],array[2],array[3],0,0,0,0])
                if clss == "Fish" :
                    truebboxes[-1][5] = 1
                elif clss == 'Flower' :
                    truebboxes[-1][6] = 1
                elif clss == 'Gravel' :
                    truebboxes[-1][7] = 1
                elif clss == 'Sugar' :
                    truebboxes[-1][8] = 1
            truebboxes = np.matrix(truebboxes) 
            show_img_with_bboxes_from_img2(img, truebboxes, bboxes)
        else :
            show_img_with_bboxes_from_img(img, bboxes)

# Fonction qui prend une image en entrée et renvoie le tenseur 25 x 9 de prédiction
def compute_y_pred_from_img(img,model, resize=(320,320)):
    im = tf.image.resize(img, resize)
    pred = model(np.array([im], dtype=np.float32))[0]
    pred = transform_netout(pred)
    return pred

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



#--------------------------------------   Méthodes d'affichage ----------------------------------------------------------


# methode qui ajuste les coordonnées pour pouvoir les afficher
def get_bbox_to_plot(im,bbox, normalised=True):
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
    x = [x1,x2,x2,x1,x1]
    y = [y1,y1,y2,y2,y1]
    return x,y


# Affichage d'une image avec une à 4 bboxes (maximum une par label) :
def show_img_with_bboxes_from_img(img, bboxes,resize=(320,320)):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    im = tf.image.resize(img, resize)
    im2 = im
    fig, ax = plt.subplots()
    im = ax.imshow(im/255)
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
        bbox = bboxes[i,1], bboxes[i,2], bboxes[i,3], bboxes[i,4] 
        x,y = get_bbox_to_plot(im2/255,bbox)  
   
    # Afficher l'image avec la bouding box    
        line, = ax.plot(x, y, "b", color = col, label = lab)
        ax.set_title('Prédiction')
        ax.set_axis_off() 
    legend = ax.legend()
    st.pyplot()

# Affichage d'une image avec groundtruth bboxes et d'une image avec predict bboxes (maximum une bbox par label)
def show_img_with_bboxes_from_img2(img, truebboxes, predbboxes, resize=(320,320)):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    im = tf.image.resize(img, resize)
    im2 = im
    im3 = im
    fig, ax = plt.subplots(1,2)
    im = ax[0].imshow(im/255)
    im2 = ax[1].imshow(im2/255)
    for j, box in enumerate([truebboxes, predbboxes]):
        for i in range(box.shape[0]):       
            if box[i,5] == 1:
                col = 'r'
                lab = 'Fish'
            if box[i,6] == 1:
                col = 'b'
                lab = 'Flower'
            if box[i,7] == 1:
                col = 'g'
                lab = 'Gravel'
            if box[i,8] == 1:
                col = 'y'
                lab = 'Sugar'
            bbox = box[i,1], box[i,2], box[i,3], box[i,4] 
            x,y = get_bbox_to_plot(im3/255,bbox)
            # Afficher l'image avec la bouding box    
            line, = ax[j].plot(x, y, "b", color = col, label = lab)
            ax[j].set_axis_off() 
            titles = ['Réalité', 'Prédiction']
            ax[j].set_title(titles[j])
    ax[0].legend()
    ax[1].legend()
    st.pyplot()



# # Calcul de l'IOU
# def list_corners_IOU(y_true, y_pred) : #y_true = list(xmin, xmax, ymin, ymax)
#     xmin = max(y_true[0], y_pred[0])
#     ymin = max(y_true[2], y_pred[2])
#     xmax = min(y_true[1], y_pred[1])
#     ymax = min(y_true[3], y_pred[3])
#     # son aire :
#     interArea = max(0,xmax-xmin+1)*max(0,ymax-ymin+1)
#     # les aires des bbox d'entrée :
#     trueArea = (y_true[1]-y_true[0]+1)*(y_true[3]-y_true[2]+1)
#     predArea = (y_pred[1]-y_pred[0]+1)*(y_pred[3]-y_pred[2]+1)
    
#     return (interArea / (trueArea + predArea - interArea))





def main():

    menu = ['Introduction', "Exploration du jeu d'entraînement", 'Détection de formes nuageuses']
    choice = st.sidebar.selectbox('Menu',menu)

    if choice == 'Détection de formes nuageuses':
        cloudDetection()

    elif choice == 'Introduction':
        welcome()

    elif choice == "Exploration du jeu d'entraînement":
        app = MultiApp()
        app.add_app("Images", Exploration_Images)
        app.add_app("Labels", Exploration_Labels)
        app.add_app("Taille des masques de formes", Exploration_Taille_Masques)
        app.add_app("Couleur des masques de formes", Exploration_Couleur_Masques)
        app.run()

if __name__ == '__main__':
    main()