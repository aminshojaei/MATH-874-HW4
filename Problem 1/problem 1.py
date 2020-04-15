# By Amin Shojaeighadikolaei
# 04/14/2020

############################################################################## Libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
       
############################################################################## Functions
def one_hot_encode(Y) :
    Output=[]
    for y in Y:
        if y == [0]:
            y = [1,0,0,0,0,0,0,0,0,0]
        elif y==[1]:
            y = [0,1,0,0,0,0,0,0,0,0]
        elif y==[2]:
            y = [0,0,1,0,0,0,0,0,0,0]
        elif y==[3]:
            y = [0,0,0,1,0,0,0,0,0,0]
        elif y==[4]:
            y = [0,0,0,0,1,0,0,0,0,0]
        elif y==[5]:
            y = [0,0,0,0,0,1,0,0,0,0]
        elif y==[6]:
            y = [0,0,0,0,0,0,1,0,0,0]
        elif y==[7]:
            y = [0,0,0,0,0,0,0,1,0,0]
        elif y==[8]:
            y = [0,0,0,0,0,0,0,0,1,0]
        elif y==[9]:
            y = [0,0,0,0,0,0,0,0,0,1]
        Output.append(y)
    return Output

############################################################################## Load datset

(images_train, labels_train), (images_test, labels_test) = tf.keras.datasets.mnist.load_data()
images_train, images_test = images_train/255.0, images_test/255.0

############################################################################## Main

mlp = tf.keras.models.Sequential()
mlp.add( tf.keras.layers.Flatten(input_shape=(28, 28)) )
mlp.add( tf.keras.layers.Dense(100, activation='relu') )
mlp.add( tf.keras.layers.Dense(100, activation='relu') )
mlp.add( tf.keras.layers.Dense(10, activation='softmax') )

label_test_1 = labels_test
labels_train = one_hot_encode(labels_train)
labels_train = np.array(labels_train)
labels_test = one_hot_encode(labels_test)
labels_test = np.array(labels_test)

mlp.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

mlp.fit(images_train, labels_train, epochs=3)

evaluation = mlp.evaluate(images_test, labels_test, verbose=2)

############################################################################## Save model & accuracy

mlp.save('first.model')
new_mlp = tf.keras.models.load_model('first.model')
predictions = new_mlp.predict(images_test)
Y_hat = np.argmax(predictions , axis = 1)
accuracy = (Y_hat == label_test_1).mean()
print(" accuracy for precition is: ", accuracy)

############################################################################## Loss

epsilon = 10e-15
Y_hat = one_hot_encode(Y_hat)
Y_hat = np.clip(Y_hat, epsilon, 1.-epsilon)
loss = - np.mean(np.log(Y_hat) * labels_test) 
print('Loss is: ',loss)

sum_score = 0.0
for i in range(len(labels_test)):
 	for j in range(len(labels_test[i])):
	   sum_score -= labels_test[i][j] * np.log(1e-15 + Y_hat[i][j])
mean_sum_score = (1.0/ len(labels_test)) * sum_score*0.1
print(mean_sum_score)








