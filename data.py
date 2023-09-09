import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

data = tf.keras.datasets.fashion_mnist

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

(train_images, train_labels), (test_images,test_labels) = data.load_data()
print(train_labels.shape)

# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.show()

train_images = train_images / 255.0
test_images = test_images / 255.0



# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
              metrics=['accuracy']
              )
model.fit(train_images, train_labels, epochs=5)

loss , acc = model.evaluate(test_images, test_labels, verbose=2)
print("\n Test Acc: ",acc)

predictions = model.predict(test_images)

# plt.figure(figsize=(10,10))
# for i in range(20):
#     plt.subplot(4,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(test_images[i], cmap=plt.cm.binary)
#     plt.xlabel('Actual: '+ class_names[test_labels[i]])
#     plt.title('Prediction: '+ class_names[np.argmax(predictions[i])])
# plt.show()

def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()






