# By Amin Shojaeighadikolaei
# 04/14/2020

############################################################################## Libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
print(tf.__version__)


############################################################################## Functions
ACCURACY_THRESHOLD = 0.99
class myCallback (tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') > ACCURACY_THRESHOLD):
            print("\nReached %2.2f%% accuracy, so stopping training!!" %(ACCURACY_THRESHOLD*100))
            self.model.stop_training = True
          
       


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


def create_model ( layers):
    model=tf.keras.models.Sequential()
    model.add( tf.keras.layers.Flatten(input_shape=(28, 28)) )
    for i, nodes in enumerate(layers):
        model.add( tf.keras.layers.Dense(nodes, activation='relu') )
    model.add( tf.keras.layers.Dense(10, activation='softmax') )
    
    model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
    
    return model
        
############################################################################## Load dataset

(images_train, labels_train), (images_test, labels_test) = tf.keras.datasets.mnist.load_data()
images_train, images_test = images_train/255.0, images_test/255.0

x_val = images_train[-15000:]
y_val = labels_train[-15000:]
images_train = images_train[:-15000]
labels_train = labels_train[:-15000]

layers = [ [5] ,[20],[200], [1000],[5000] ]
callbacks = myCallback()
LOSS = []

############################################################################## Main I ( for one hidden layer)
test_loss_total_1=[]
test_accuracy_total_1=[]
for i in layers:
    mlp= create_model(i)
    print('Simulation for '+str(i)+' layer: ')
    
 
    history= mlp.fit(images_train, labels_train,validation_data=(x_val,y_val), epochs=10,verbose=2,callbacks=[callbacks])
    
    # convert the history.history dict to a pandas DataFrame:     
    hist_df = pd.DataFrame(history.history)
    loss=hist_df.to_numpy()
    LOSS.append(loss)
    Looss= np.array(LOSS)
    
    test_loss_1 , test_accuracy_1 = mlp.evaluate(images_test, labels_test, verbose=1)
    test_loss_total_1.append(test_loss_1)
    test_accuracy_total_1.append(test_accuracy_1)


############################################################################## Plot( For one hidden layer) 

final_loss=[[] for i in range(len(layers))]
final_val_loss=[[] for i in range(len(layers))]
for i in range(len(layers)):
    for j in range(Looss[i].shape[0]):
        final_loss[i].append(Looss[i][j][0])
        final_val_loss[i].append(Looss[i][j][2])

fig, axs= plt.subplots(5,2 , figsize=(10,10))
fig.suptitle('LOSS')
plt.xlabel('Epochs count')
for i in range(len(layers)):
    
    axs[i,0].plot(final_loss[i],label='loss '+str(layers[i])+'layer')
    axs[i,0].legend(fontsize=14)
    axs[i,1].plot(final_val_loss[i],label='val loss '+str(layers[i])+'layer')
    axs[i,1].legend(fontsize=14)
#     axs[i,0].set_yticks(np.arange(0, 0.5, step=0.1))
#     axs[i,1].set_yticks(np.arange(0, 0.5, step=0.1))
plt.savefig('comparison for one hidden layer.png')
plt.show()


############################################################################## Main II (for two hidden layer)


layers = [ [5,5] ,[20,20],[100,100], [500,500],[1500,1500] ]
callbacks = myCallback()
LOSS = []
test_loss_total_2=[]
test_accuracy_total_2=[]
for i in layers:
    mlp= create_model(i)
    print('Simulation for '+str(i)+' layer: ')
    
 
    history= mlp.fit(images_train, labels_train,validation_data=(x_val,y_val), epochs=10,verbose=2,callbacks=[callbacks])
    
    # convert the history.history dict to a pandas DataFrame:     
    hist_df = pd.DataFrame(history.history)
    loss=hist_df.to_numpy()
    LOSS.append(loss)
    
    test_loss_2 , test_accuracy_2 = mlp.evaluate(images_test, labels_test, verbose=1)
    test_loss_total_2.append(test_loss_2)
    test_accuracy_total_2.append(test_accuracy_2)
    


############################################################################## Plot(for two hidden layer)


# LOSS
# loss.shape
Looss= np.array(LOSS)
final_loss=[[] for i in range(len(layers))]
final_val_loss=[[] for i in range(len(layers))]
for i in range(len(layers)):
    for j in range(Looss[i].shape[0]):
        final_loss[i].append(Looss[i][j][0])
        final_val_loss[i].append(Looss[i][j][2])
   
fig, axs= plt.subplots(5,2 , figsize=(10,10))
fig.suptitle('LOSS')

plt.xlabel('Epochs count')
for i in range(len(layers)):
    
    axs[i,0].plot(final_loss[i],label='loss '+str(layers[i])+'layer')
    axs[i,0].legend(fontsize=14)
    axs[i,1].plot(final_val_loss[i],label='val loss '+str(layers[i])+'layer')
    axs[i,1].legend(fontsize=14)

plt.savefig('comparison for two hidden layer.png')
plt.show()

############################################################################## Plot compare 2 optimal
one_hidden_layer=['5','20','200','1000','5000']
thow_hidden_layer=['[5,5]','[20,20]','[100,100]','[200,200]','[1000,1000]']
fig, axs= plt.subplots(2,2, figsize=(10,10))
fig.suptitle('LOSS-Test')


axs[0,0].plot(one_hidden_layer,test_loss_total_1, color='red',label='test_loss-1 hidden')
axs[0,0].legend()
axs[1,0].plot(one_hidden_layer,test_accuracy_total_1, color='red',label='accuracy-1 hidden')
axs[1,0].legend()


axs[0,1].plot(thow_hidden_layer,test_loss_total_2, color='blue',label='test_loss-2 hidden')
axs[0,1].legend()
axs[1,1].plot(thow_hidden_layer,test_accuracy_total_2, color='blue',label='accuracy-2 hidden')
axs[1,1].legend()

plt.xlabel('Number of hidden layer')
plt.savefig('comparison for test.png')
plt.show()



  
