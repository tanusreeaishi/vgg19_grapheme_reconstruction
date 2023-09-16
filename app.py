import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import keras
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import *
from keras.models import Sequential, Model, load_model
import streamlit as st
from tensorflow.keras import mixed_precision
import keras.backend as K
import pandas as pd
from tensorflow.keras import mixed_precision
#import matplotlib.pyplot as plt
#import splitfolders
import zipfile
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, LearningRateScheduler
import shutil
import sys
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import locale
from keras.preprocessing.image import ImageDataGenerator
# locale.setlocale(locale.LC_ALL, 'bn_BD.UTF-8')
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
import seaborn as sn
from io import StringIO
from PIL import Image
from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt

st.title('Grapheme Reconstruction :blue[(VGG19)]')
st.subheader('Upload an image and get the Grapheme Reconstructed', divider = 'rainbow')
uploaded_file = st.file_uploader("Choose a file")

df = pd.read_csv('train.csv')

grapheme_roots = df['grapheme_root'].values
n, c = np.unique(grapheme_roots, return_counts=True)
total_grapheme_roots = len(n)
# print(total_grapheme_roots, 'total_grapheme_roots')


vowel_diacritic = df['vowel_diacritic'].values
n, c = np.unique(vowel_diacritic, return_counts=True)
total_vowel_diacritic = len(n)
# print(total_vowel_diacritic, 'total_vowel_diacritic')


consonant_diacritic = df['consonant_diacritic'].values
n, c = np.unique(consonant_diacritic, return_counts=True)
total_consonant_diacritic = len(n)
# print(total_consonant_diacritic, 'total_consonant_diacritic')

vgg19 = VGG19(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3),
    pooling=None,
)

# Global Average Pooling
vgg19_gap = GlobalAveragePooling2D()(vgg19.output)

# Dense layers for each output
d_grapheme_root = Dense(512, activation='relu')(vgg19_gap)
d_grapheme_root = Dropout(0.5)(d_grapheme_root)
d_grapheme_root = Dense(256, activation='relu')(d_grapheme_root)

d_vowel_diacritic = Dense(512, activation='relu')(vgg19_gap)
d_vowel_diacritic = Dropout(0.5)(d_vowel_diacritic)
d_vowel_diacritic = Dense(256, activation='relu')(d_vowel_diacritic)

d_consonant_diacritic = Dense(512, activation='relu')(vgg19_gap)
d_consonant_diacritic = Dropout(0.5)(d_consonant_diacritic)
d_consonant_diacritic = Dense(256, activation='relu')(d_consonant_diacritic)

# Output layers with softmax activation for each task
grapheme_root = Dense(total_grapheme_roots, activation='softmax', name='grapheme_root')(d_grapheme_root)
vowel_diacritic = Dense(total_vowel_diacritic, activation='softmax', name='vowel_diacritic')(d_vowel_diacritic)
consonant_diacritic = Dense(total_consonant_diacritic, activation='softmax', name='consonant_diacritic')(d_consonant_diacritic)

# Create the final model
model = Model(inputs=vgg19.inputs, outputs=[grapheme_root, vowel_diacritic, consonant_diacritic])

# Display the model summary
# model.summary()

# FOR RECONSTRUCTION PURPOSE
df_gr = pd.read_csv('train.csv')
df_gr = df_gr.drop_duplicates(subset=['grapheme'])
dv = df_gr.values
grapheme_roots = []
_grapheme_roots = []

for v in dv:
    if v[1] != 0 and v[2] == 0 and v[3] == 0 and v[-1] not in _grapheme_roots:
        _grapheme_roots.append(v[-1])
        grapheme_roots.append({'numeric': v[1], 'value': v[-1]})
        
vowel_diacritics = {}
vowel_diacritics[0] = ''
vowel_diacritics[1] = 'া'
vowel_diacritics[2] = 'ি'
vowel_diacritics[3] = 'ী'
vowel_diacritics[4] = 'ু'
vowel_diacritics[5] = 'ূ'
vowel_diacritics[6] = 'ৃ'
vowel_diacritics[7] = 'ে'
vowel_diacritics[8] = 'ৈ'
vowel_diacritics[9] = 'ো'
vowel_diacritics[10] = 'ৌ'


consonant_diacritics = {}
consonant_diacritics[0] = ''
consonant_diacritics[1] = 'ঁ'
consonant_diacritics[2] = '\u09b0\u09cd'
consonant_diacritics[3] = '্য' #//ref + ja fala
consonant_diacritics[4] = '্য'
consonant_diacritics[5] = '্র'
consonant_diacritics[6] = '্র্য'
consonant_diacritics[7] = '্র' #ref + ra fala


def get_grapheme_root(numeric):
    for item in grapheme_roots:
        if item['numeric'] == numeric:
            return item['value']
    return ''

def get_vowel_diacritic(numeric):
    global vowel_diacritics
    return vowel_diacritics[numeric]

def get_consonant_diacritic(numeric):
    global consonant_diacritics
    return consonant_diacritics[numeric]
  
consonant_middle=[5,4,6]
consonant_after=[1]
consonant_before=[2]
consonant_combined=[3,7]
def get_grapheme(gr,vd,cd):
    consonant_middle=[5,4,6]
    consonant_after=[1]
    consonant_before=[2]
    consonant_combined=[3,7]
 
    if cd==0:
        return get_grapheme_root(gr)+get_vowel_diacritic(vd)
    elif cd in consonant_middle:
        return get_grapheme_root(gr)+get_consonant_diacritic(cd)+get_vowel_diacritic(vd)
    elif cd in consonant_before: #ref
        return get_consonant_diacritic(cd)+get_grapheme_root(gr)+get_vowel_diacritic(vd)
    elif cd in consonant_combined :#ref+ ja fala
        
        return '\u09b0\u09cd'+get_grapheme_root(gr)+get_consonant_diacritic(cd)+get_vowel_diacritic(vd)
    
    elif cd in consonant_after:
        return get_grapheme_root(gr)+get_vowel_diacritic(vd)+get_consonant_diacritic(cd)
    

def get_grapheme_root(numeric):
    for item in grapheme_roots:
        if item['numeric'] == numeric:
            return item['value']
    return ''

def get_vowel_diacritic(numeric):
    global vowel_diacritics
    return vowel_diacritics[numeric]

def get_consonant_diacritic(numeric):
    global consonant_diacritics
    return consonant_diacritics[numeric]

consonant_middle=[5,4,6]
consonant_after=[1]
consonant_before=[2]
consonant_combined=[3,7]
def get_grapheme(gr,vd,cd):
    consonant_middle=[5,4,6]
    consonant_after=[1]
    consonant_before=[2]
    consonant_combined=[3,7]

    if cd==0:
        return get_grapheme_root(gr)+get_vowel_diacritic(vd)
    elif cd in consonant_middle:
        return get_grapheme_root(gr)+get_consonant_diacritic(cd)+get_vowel_diacritic(vd)
    elif cd in consonant_before: #ref
        return get_consonant_diacritic(cd)+get_grapheme_root(gr)+get_vowel_diacritic(vd)
    elif cd in consonant_combined :#ref+ ja fala

        return '\u09b0\u09cd'+get_grapheme_root(gr)+get_consonant_diacritic(cd)+get_vowel_diacritic(vd)

    elif cd in consonant_after:
        return get_grapheme_root(gr)+get_vowel_diacritic(vd)+get_consonant_diacritic(cd)


hf_hub_download(repo_id="samanjoy2/vgg19_banglagrapheme", filename="testvgg.hdf5", local_dir = './')

model.load_weights('testvgg.hdf5')

if uploaded_file:

    image = Image.open(uploaded_file)
    
    # st.image(image,width=200, caption='Image Used for Reconstruction')

    y_true_grapheme_root_new = []
    y_true_vowel_diacritic_new = []
    y_true_consonant_diacritic_new = []

    y_pred_grapheme_root_new = []
    y_pred_vowel_diacritic_new = []
    y_pred_consonant_diacritic_new = []

    img = tf.keras.utils.load_img(uploaded_file, color_mode='rgb',target_size=(224, 224))
    img = tf.keras.utils.img_to_array(img)/255.
    img = np.expand_dims(img, axis=0)
    pr = model.predict(img, verbose=0)

    pred_grapheme_root = np.argmax(pr[0], axis=-1)[0]
    pred_vowel_diacritic = np.argmax(pr[1], axis=-1)[0]
    pred_consonant_diacritic = np.argmax(pr[2], axis=-1)[0]

    y_pred_grapheme_root_new.append(pred_grapheme_root)
    y_pred_vowel_diacritic_new.append(pred_vowel_diacritic)
    y_pred_consonant_diacritic_new.append(np.argmax(pr[2], axis=-1)[0])

    plt.imshow(img[0, ::])
    st.pyplot(plt, use_container_width=True)
    
    st.markdown('Grapheme Root: '+str(get_grapheme_root(pred_grapheme_root)))
    st.markdown('Vowel Diacritic: '+str(get_vowel_diacritic(pred_vowel_diacritic)))
    st.markdown('Constant Diacritic: '+str(get_consonant_diacritic(np.argmax(pr[2], axis=-1)[0])))

    # st.subheader("Grapheme : "+str(get_grapheme(pred_grapheme_root, pred_vowel_diacritic, pred_consonant_diacritic)))
    st.subheader(":green[Main Prediction =] "+str(get_grapheme(pred_grapheme_root, pred_vowel_diacritic, pred_consonant_diacritic)))
