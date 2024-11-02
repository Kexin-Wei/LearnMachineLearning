# Tensorflow 2 Intro
import tensorflow as tf
# tf.__version__
import matplotlib.pyplot as plt

# Get Dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()


# Get the data shape
print(x_train.shape)
print(y_train.shape)
N_INPUT = x_train.shape[1]*x_train.shape[2]


# Take a look into the data
#plt.imshow(x_train[0],cmap = plt.cm.binary)
#plt.show()

# Data Modify
x_train = tf.keras.utils.normalize(x_train)
x_test  = tf.keras.utils.normalize(x_test)

#plt.imshow(x_train[0],cmap = plt.cm.binary)
#plt.show()

# Build the train model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten()) # input layer
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) # hidden layer 1
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) # hidden layer 2
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax)) # output layer

model.compile(
    optimizer= "Adam",
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train,y_train, epochs=3)

val_loss,val_acc = model.evaluate(x_test,y_test)