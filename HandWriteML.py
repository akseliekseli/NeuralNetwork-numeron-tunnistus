import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Luetaan data: Data on sisäänrakennettuna keras-kirjaston datasetteihin.
# x-alkiot ovat 28x28 matriisi grayscalena. y-alkiot ovat numero, jota matriisi esittää
(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()


# Tästä kommentit poistamalla saa plotattua opetusdatan 9 ensimmäistä kuvaa

#fig = plt.figure()
#for i in range(9):
  #plt.subplot(3,3,i+1)
  #plt.tight_layout()
  #plt.imshow(X_train[i], cmap='gray', interpolation='none')
  #plt.title("Digit: {}".format(Y_train[i]))
  #plt.xticks([])
  #plt.yticks([])
#plt.show()





X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)


# Luodaan neuroverkko, jossa on kaksi piilokerrosta.
# Aktivointifunktioina ReLU (Rectified linear unit) ja Softmax

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))

model.compile( optimizer = 'adam', 
            loss = 'sparse_categorical_crossentropy',
            metrics = ['accuracy'])

# Ajetaan opetusdata neljä kertaa neuroverkon läpi
model.fit(X_train, Y_train, epochs = 4)

#Lasketaan loss ja accuracy
val_loss, val_acc = model.evaluate(X_test,  Y_test)
print(val_loss, val_acc)

# Tallennetaan malli, jotta sitä voidaan käyttää sovelluksessa.
model.save('mnistNN.model')
