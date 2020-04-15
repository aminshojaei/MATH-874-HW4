# By Amin Shojaeighadikolaei
# 04/14/2020

############################################################################## Libraries

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

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
    return np.asarray(Output)


class MNISTmodel(tf.keras.Model):
    def __init__(self):
        super(MNISTmodel, self).__init__()
        self.layer0 = tf.keras.layers.Flatten()
        self.layer1 = tf.keras.layers.Dense(40, activation='relu')
        self.layer2 = tf.keras.layers.Dense(40, activation='relu')
        self.layer3 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, X):
        X1 = self.layer0(X)
        X2 = self.layer1(X1)
        X3 = self.layer2(X2)
        Y = self.layer3(X3)
        return Y


def iterate(images, labels):
    with tf.GradientTape() as tape:
        predictions = mlp(images)
        l = loss(labels, predictions)
    gradients = tape.gradient(l, mlp.trainable_variables)
    op.apply_gradients(zip(gradients, mlp.trainable_variables))

    mlp_loss(l)
    mlp_accuracy(labels, predictions)
    

def validation(data):
    loss=[]
    epsilon = 10e-15
    for sample in data:
        prediction = mlp(sample['image'])
        Y_hat = prediction
        Y_hat = np.clip(Y_hat, epsilon, 1.-epsilon)
        cross = - np.log(Y_hat) * one_hot_encode(sample['label'])
        los = np.mean(np.sum(cross,axis=1))
        loss.append(los)
        prediction = np.argmax(prediction, axis=0)
        accuracy = np.mean(prediction == one_hot_encode(sample['label']))
        
    return np.mean(loss) , np.mean(accuracy)    
    
############################################################################## Load dataset

il_train = tfds.load('mnist', shuffle_files=True, batch_size=32)
train = il_train['train'].take(int(0.75 * 60000))
val = il_train['train'].take(int(0.25 * 60000))
test = il_train['test']

############################################################################## Main


mlp = MNISTmodel()
mlp_loss = tf.keras.metrics.Mean()
mlp_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()


loss = tf.keras.losses.SparseCategoricalCrossentropy()
op = tf.keras.optimizers.Adam()


for epoch in range(2):
    mlp_loss.reset_states()
    mlp_accuracy.reset_states()

    for example in train:
        iterate(example['image'], example['label'])
        
    validation_loss, accuracy = validation(val)

    print('Epoch {}/2 - train loss: {} - train accuracy: {}, validation loss {} validation accuraccy: {}'.format(
        epoch + 1, mlp_loss.result(), mlp_accuracy.result(), validation_loss, accuracy))


test_loss, accuracy = validation(test)
print('train loss: {} - train accuracy: {}, test loss {} test accuraccy: {}'.format(
    mlp_loss.result(), mlp_accuracy.result(), test_loss, accuracy))

































    