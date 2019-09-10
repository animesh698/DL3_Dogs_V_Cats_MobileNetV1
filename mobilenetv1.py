#mobilenetV1
import numpy as np
import keras
from keras import backend as K
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

train_path='C:/Users/legend698/Desktop/train'
test_path='C:/Users/legend698/Desktop/ANIMESH/python/Cats-and-Dogs/test'
valid_path='C:/Users/legend698/Desktop/valid'

#preprocessing_funtion is used in the ImageDataGenerator Funtion,it does the necessary preprocessing on the images obtained from flow_from_directory
train_batches=ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input).flow_from_directory(train_path,target_size=(224,224),batch_size=10)
test_batches =ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input).flow_from_directory(test_path,target_size=(224,224),classes=None,class_mode=None,batch_size=10,shuffle=False)
valid_batches=ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input).flow_from_directory(valid_path,target_size=(224,224),batch_size=10)

#Importing the mdodel
mobile= keras.applications.mobilenet.MobileNet()
# model.summary()

#We will grab the o/p from the 6th to the last layer and store it in a var
x=mobile.layers[-6].output
prediction=Dense(2,activation='softmax')(x)	
#Model is a constuctor called by the Kera functional API
#We create an instance of the model class
#Format diff from seq model,here we specify the inputs to be the input of the OG mobilnet and o/ps what we created in the predictions var 
model=Model(inputs=mobile.input,outputs=prediction)
#We will freeze the weights of the 1st 5 layers 
for layer in model.layers[:-5]:
	layer.trainable=False

#Train the model
model.compile(Adam(lr=.0001),loss='categorical_crossentropy',metrics=['accuracy'])
model.fit_generator(train_batches,steps_per_epoch=2500,validation_data=valid_batches,validation_steps=40,epochs=4,verbose=2)

#predict
test_labels=test_batches.classes
print(test_labels)
# print(test_batches.class_indices)
predict=model.predict_generator(test_batches,verbose=0,steps=700)
cm=confusion_matrix(test_labels,predict.argmax(axis=1))
#argmax predicts only the higher prediction of the 2(or more) classes involved

def plot_confusion_matrix(cm,classes,normalize=False,title='Confusion Matrix',cmap=plt.cm.Blues):
	plt.imshow(cm,interpolation='nearest',cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks=np.arange(len(classes))
	plt.xticks(tick_marks,classes,rotation=45)
	plt.yticks(tick_marks,classes)

	if normalize:
		cm=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion Matrix,without normalization')

	print(cm)

	thresh = cm.max() / 2.
	for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
		plt.text(j,i,cm[i,j],horizontalalignment="center",color="white" if cm[i,j]>thresh else "black")
	plt.tight_layout()
	plt.ylabel('True Label')
	plt.xlabel('Predicted Label')
cm_plot_lables=['cat','dog']
plot_confusion_matrix(cm,cm_plot_lables,title='Confusion Matrix')
plt.show()
model.save('mbnet.h5')