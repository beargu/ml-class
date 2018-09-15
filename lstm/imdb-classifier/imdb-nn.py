import imdb
import numpy as np
from keras.preprocessing import text
import wandb
from sklearn.linear_model import LogisticRegression


from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.utils import np_utils

import wandb
from wandb.keras import WandbCallback

wandb.init()
config = wandb.config
config.vocab_size = 1000

config.dropout = 0.5 # 0.2
config.dense_layer_size = 128
config.epochs = 10


(X_train, y_train), (X_test, y_test) = imdb.load_imdb()
# up to here, X_train is text



#print(X_train.shape, X_test.shape)

tokenizer = text.Tokenizer(num_words=config.vocab_size)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_matrix(X_train)
X_test = tokenizer.texts_to_matrix(X_test)


# normalization
#X_train = X_train.astype("float32")/255.0
#X_test = X_test.astype("float32")/255.0

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
labels = range(10)

num_classes = y_train.shape[1]

print(X_train.shape, X_test.shape, num_classes)

#exit()

# create model
model=Sequential()
#no need for this, input is alreay a vector, model.add(Flatten(input_shape=(X_train.shape[1],1)))
#model.add(Dense(num_classes, activation="softmax"))
model.add(Dense(150, activation="linear"))
model.add(Dropout(0.2))

model.add(Dense(150, activation="sigmoid"))
model.add(Dropout(0.2))


model.add(Dense(num_classes, activation="sigmoid"))


model.compile(loss='categorical_crossentropy', optimizer='adam',                metrics=['accuracy'])


# Fit the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test),
                    callbacks=[WandbCallback(save_model=False)])
model.save('mode_imdb.h3')
print(model.predict(X_test[1:10]))





#bow_model = LogisticRegression() # from scikit-learn
#bow_model.fit(X_train, y_train)#

#pred_train = bow_model.predict(X_train)
#acc = np.sum(pred_train==y_train)/len(pred_train)

#pred_test = bow_model.predict(X_test)
#val_acc = np.sum(pred_test==y_test)/len(pred_test)
#wandb.log({"val_acc": val_acc, "acc": acc})