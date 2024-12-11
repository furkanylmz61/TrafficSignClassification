import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

# Veri ve etiketlerin depolanacağı listeler
data = []
labels = []
classes = 43
cur_path = os.getcwd()

# Görüntüleri ve etiketlerini yükleme
for i in range(classes):
    path = os.path.join(cur_path, 'train', str(i))
    if not os.path.exists(path):
        print(f"Class {i} folder not found at path: {path}")
        continue

    images = os.listdir(path)

    for a in images:
        try:
            image = Image.open(os.path.join(path, a))
            image = image.resize((30, 30))
            image = np.array(image)
            data.append(image)
            labels.append(i)
        except Exception as e:
            print(f"Error loading image {a} in class {i}: {e}")

# Listeleri numpy array'lere dönüştürme
data = np.array(data)
labels = np.array(labels)

# Boş veri kontrolü
if len(data) == 0 or len(labels) == 0:
    raise ValueError("Data or labels are empty. Check the dataset path and structure.")

print(data.shape, labels.shape)

# Eğitim ve test veri setlerini ayırma
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Etiketleri one-hot encoding'e dönüştürme
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)

# Modelin oluşturulması
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))

# Modelin derlenmesi
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Modelin eğitilmesi
epochs = 15
history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))
model.save("my_model.h5")

# Doğruluk grafikleri
plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

# Test verisi doğruluğu
from sklearn.metrics import accuracy_score

# Test CSV dosyasını yükleme
y_test_csv = pd.read_csv('Test.csv')
labels_csv = y_test_csv["ClassId"].values
imgs_csv = y_test_csv["Path"].values

data = []

for img in imgs_csv:
    try:
        image = Image.open(img)
        image = image.resize((30, 30))
        data.append(np.array(image))
    except Exception as e:
        print(f"Error loading image {img}: {e}")

X_test = np.array(data)

# Tahminler
pred = model.predict(X_test)
pred_classes = np.argmax(pred, axis=1)

# Doğruluk
print(accuracy_score(labels_csv, pred_classes))
