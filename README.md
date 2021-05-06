README
Creation of ipynb file
•	First jupyter notebook was launched with the command Jupyter Notebook from Anaconda Prompt.
•	Then a new ipynb file or Python 3 file was created.
•	Next the codes were coded accordingly.
Execution
First, we need to extract the dataset and train the data model.
def make_train_data(label,DIR):
    for img in tqdm(os.listdir(DIR)):
        path = os.path.join(DIR,img)
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        
        X.append(np.array(img))
        Z.append(str(label))
X=[]
Z=[]
IMG_SIZE=150
NOTFIRE='Test/No_Fire'
FIRE='Test/Fire'

make_train_data('NOTFIRE',NOTFIRE)
make_train_data('FIRE',FIRE)
The images are imported into the training data from the respective folders.

The next step is of subplotting the images and training them accordingly to predict or identify the fire images and not fire images
Then we load the ResNet50 model and use transfer learning to use the prebuilt trained models to use as a prediction system.
base_model=ResNet50(include_top=False, weights='imagenet',input_shape=(150,150,3), pooling='max')
base_model.summary()
model=Sequential()
model.add(base_model)
model.add(Dropout(0.20))
model.add(Dense(2048,activation='relu'))
model.add(Dense(1024,activation='relu'))
model.add(Dense(512,activation='relu'))
model.add(Dense(2,activation='softmax'))
epochs=100
batch_size=128
red_lr=ReduceLROnPlateau(monitor='val_acc', factor=0.1, min_delta=0.0001, patience=2, verbose=1)
base_model.trainable=True # setting the VGG model to be trainable.
model.compile(optimizer=Adam(lr=1e-5),loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
Then we store the model in a history variable
History = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test,y_test))
Then we plot the epochs vs accuracy and Model Loss graphs.
plt.plot(History.history['accuracy'])
plt.plot(History.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()
 

plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()

 


