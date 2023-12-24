import speech_recognition as sr 
import time

import numpy as np

import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
import nltk
import re
import keras
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Dropout
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
#import tensorflow
from warnings import filterwarnings
filterwarnings('ignore')

import pickle

from pocketsphinx import LiveSpeech


class Voice2Intent :
    def __init__(self,word_tokenizer_path=None,unique_intent_path=None,max_length_path=None,model_path=None) :
        self.word_tokenizer_path = word_tokenizer_path #'word_tokenizer_2.pkl'
        self.unique_intent_path = unique_intent_path #'unique_intent_2.pkl'
        self.max_length_path = max_length_path #'max_length_2.pkl'
        self.model_path = model_path #"Chat_Model_2.h5"


    def LoadData(self) :

        print('Data Loading...')
        with open(self.word_tokenizer_path,'rb') as f :
            self.word_tokenizer = pickle.load(f)

        with open(self.unique_intent_path,'rb') as f :
            self.unique_intent = pickle.load(f)

        with open(self.max_length_path,'rb') as f :
            self.max_length = pickle.load(f)


    def RecSpeech(self) :
        recognizer = sr.Recognizer()
        text = None

        with sr.Microphone() as source :

            print('Please say something !!! ... ')
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source , timeout = 3)

            try :
                print("Recognizing ... ")
                # text = recognizer.recognize_sphinx(audio)
                text = recognizer.recognize_google(audio)
                # text = recognizer.recognize_tensorflow(audio)
                print('The output : ',text)
            except sr.UnknownValueError:
                print("Could not understand audio.")
            except sr.RequestError as e:
                print(f"Error with the API request; {e}")

        return text

    def padding_doc(self,encoded_doc, max_length):
        return(pad_sequences(encoded_doc, maxlen = max_length, padding = "post"))

    def predictions(self,text):
        clean = re.sub(r'[^ a-z A-Z 0-9]', " ", text)
        test_word = word_tokenize(clean)
        test_word = [w.lower() for w in test_word]
        test_ls = self.word_tokenizer.texts_to_sequences(test_word)
        # print(test_word)
        #Check for unknown words
        if [] in test_ls:
            test_ls = list(filter(None, test_ls))
        test_ls = np.array(test_ls).reshape(1, len(test_ls))
        x = self.padding_doc(test_ls, self.max_length)
        pred = self.model.predict(x)
        
        return pred

    def get_final_output(self,pred, classes):
        predictions = pred[0]
        
        classes = np.array(classes)
        ids = np.argsort(-predictions)
        classes = classes[ids]
        predictions = -np.sort(-predictions)
        
        return classes

    def Text2Intent(self) :
        text = self.RecSpeech()
        self.model = load_model(self.model_path)
        pred = self.predictions(text)
        res = self.get_final_output(pred , self.unique_intent)[0]
        print('Intent : ',res)
        return res


def main() :
    word_tokenizer_path = 'Speech2Intent/word_tokenizer_2.pkl'
    unique_intent_path = 'Speech2Intent/unique_intent_2.pkl'
    max_length_path = 'Speech2Intent/max_length_2.pkl'
    model_path = "Speech2Intent/Chat_Model_2.h5"
    vi = Voice2Intent(word_tokenizer_path,unique_intent_path,max_length_path,model_path)
    vi.LoadData()
    # text = vi.RecSpeech() 
    print(vi.Text2Intent())

if __name__ == "__main__" :
    main()