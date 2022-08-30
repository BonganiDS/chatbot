#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 10:44:41 2022

@author: bonganisiamtinta
"""

## importing libraries##
import tensorflow
import nltk
import random
import json
import pickle
from nltk.stem import WordNetLemmatizer
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD


lemmatizer = WordNetLemmatizer()

## Declaring Constants ##
words = []
labels = []
docs = []
ignore_list = ['?' , '!']

 ## Loading datasets
 
dataset = open('intents.json').read()
intents = json.loads(dataset)

#Preprocessing Data ##

for intent in intents['intents']:
    for pattern in intent['patterns']:
        
        #tokenize each word#
        word_token = nltk.word_tokenize(pattern)
        words.extend(word_token)
        #add documents in the corpus 
        docs.append((word_token,intent['tag']))
        
        #add to our labels list 
        if intent['tag'] not in labels:
            labels.append(intent['tag'])
            
# lemmatize each word and sort words by removing duplicates :#
            word = [lemmatizer.lemmatize(word.lower()) for word in  words if word not in ignore_list]
            words = sorted(list(set(word)))
            labels = sorted(list(set(labels)))
     
 ## saving words and labels ##
 
            pickle.dump(words,open('words.pkl','wb') )
            pickle.dump(labels,open('labels.pkl','wb') )
            
## Creating our Training data ##

training_data = []

output = [0]*len(labels)

for doc in docs:
    bag_of_words = []
    pattern_words = doc[0]
    
    #lemmatize pattern words#
    
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    
    for w in words:
        if w in pattern_words:
            bag_of_words.append(1)
        else:
            bag_of_words.append(0)
            
    output_row = list(output)
    output_row[labels.index(doc[1])] = 1
    
training_data.append([bag_of_words,output_row])

##Shuffle and converting ##

random.shuffle(training_data)
training_data = np.array(training_data)

#Splitting the data into x_train and y_train
x_train = list(training_data[:, 0])
y_train = list(training_data[:, 1])

#Model Creation

model = Sequential()
model.add(Dense(128, input_shape = (len(x_train[0]),),activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]),activation = 'softmax'))



#compile and fit our model to find the accuracy

sgd_optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd_optimizer, metrics=['accuracy'])

# fit the model

history = model.fit(np.array(x_train), np.array(y_train), epochs = 200, batch_size = 5, verbose =1)

#saving model
model.save('chatbot.h5', history)
model.summary()

print("model complete")

