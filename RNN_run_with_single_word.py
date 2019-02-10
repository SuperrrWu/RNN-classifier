import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dropout,Dense
from keras.layers import Activation
from keras.layers.embeddings import Embedding
#from keras.layers import LSTM
from keras.layers.recurrent import SimpleRNN
import numpy as np
import jieba
str_list=[]
path="C:/Users/Oliver/Desktop/RNN/"
for i in os.listdir(path+"唐诗"):
    with open(path+"唐诗/"+str(i)) as f:
        data=f.readline()
        data=list(data)
        str_list_pre=" ".join(data)
        str_list.append(data)
for i in os.listdir(path+"宋词"):
    with open(path+"宋词/"+str(i)) as f:
        data=f.readline()
        data=list(data)
        str_list_pre=" ".join(data)
        str_list.append(data)
token=Tokenizer(num_words=5000)
token.fit_on_texts(str_list)
print(token.word_index)
X_train_seq=token.texts_to_sequences(str_list)
X_train=sequence.pad_sequences(X_train_seq,maxlen=40,padding="post")
Y_train=[0]*1191+[1]*1186
print(len(X_train))
print(len(Y_train))
diction={"唐诗":0,"宋词":1}
model=Sequential()
model.add(Embedding(
    output_dim=16,
    input_dim=5000,
    input_length=40
))
model.add(Dropout(0.1))
model.add(SimpleRNN(units=16))
model.add(Dense(32))
model.add(Activation("relu"))
model.add(Dropout(0.1))
model.add(Dense(16))
model.add(Activation("relu"))
model.add(Dropout(0.1))
model.add(Dense(1))
model.add(Activation("sigmoid"))
model.compile(loss="binary_crossentropy",optimizer="adam",metrics=['accuracy'])
model.fit(X_train,Y_train,verbose=1,epochs=100,batch_size=40)
print(X_train)
new_list=[]
with open("C:/Users/Oliver/Desktop/RNN/test/test1.txt","r") as f:
    data=f.readline()
    data=list(data)
    data=" ".join(data)
    new_list.append(data)
    f.close()
print(new_list)
X_test=token.texts_to_sequences(new_list)
X_test=sequence.pad_sequences(X_test,maxlen=40,padding="post")
print(token.word_index)
print(X_test)
print(X_train[0])
b=model.predict_classes(X_test).astype("int")
result=np.sum(b)
diction_list=list(diction)
print(diction_list[result])