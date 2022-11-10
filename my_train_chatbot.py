from xml.dom.minidom import Document
from xml.dom.xmlbuilder import DocumentLS
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
Lemmatizer=WordNetLemmatizer()
import json
import pickle
import numpy as np
import keras
import random


words=[]
classes=[]
document=[]
ignore_words=['?','!']
data_file=open('C:/Users/shiva/Desktop/codes/python/PROJECT/AI CHATBOT/Conversation.json','r')
intents=json.load(data_file)


for intent in intents['intents']:    
    for pattern in intent['patterns']:
        w=nltk.word_tokenize(pattern)
        words.extend(w)
        document.append((w,intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])


words=[Lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words=sorted(list(set(words)))#used type cast in set to remove repeating words from the new list
classes=sorted(list(set(classes)))
print(len(document),"documents")
print(len(classes),"classes")
print(len(words),"unique lemmatized words")


pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))


training=[]
output_empty=[0]*len(classes)#len of classes times 0
for doc in document:
    bag=[]
    pattern_words=doc[0]
    pattern_words=[Lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    output_row=list(output_empty)    
    output_row[classes.index(doc[1])]=1    
    training.append([bag,output_row])


random.shuffle(training)
training=np.array(training)
train_x=list(training[:,0])
train_y=list(training[:,1])
print("this is data labels",train_y)

model=keras.models.Sequential()
model.add(keras.layers.Dense(128,input_shape=(len(train_x[0]),),activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(64,activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(len(train_y[0]),activation='softmax'))
sgd=keras.optimizers.SGD(learning_rate=0.01,decay=1e-6,momentum=0.9,nesterov=True)#momentum:float hyperparameter >= 0 that accelerates gradient descent in the relevant direction and dampens oscillations.
#netrov is the type of momentum
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
hist=model.fit(np.array(train_x),np.array(train_y),epochs=200,batch_size=5,verbose=1)#verbose retuens  a regex which will retrun the given string
model.save('chatbot_model.h5',hist)
print('model created')




