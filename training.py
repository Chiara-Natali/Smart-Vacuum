!pip install -U tensorflow-datasets
import tensorflow_datasets as tfds
import numpy as np
import os
import sys
import time
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

mappa_emnist = {
    'S': 19, 'F': 6, 'D': 4, 'V': 22, 'X': 24, 'C': 3
}
classi_volute = list(mappa_emnist.keys())
indici_voluti = list(mappa_emnist.values())

print("Scaricamento e caricamento eMNIST in corso...")
ds_train, ds_test = tfds.load(
    'emnist/letters',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=False
)

def ds_to_numpy(ds):
    images, labels = [], []
    for img, lbl in ds.as_numpy_iterator():
        images.append(img)
        labels.append(lbl)
    return np.array(images), np.array(labels)

x_raw_train, y_raw_train = ds_to_numpy(ds_train)
x_raw_test, y_raw_test = ds_to_numpy(ds_test)

def fix_emnist_axes(images):
    return np.array([np.transpose(img, (1, 0, 2)) for img in images])

print("Correzione orientamento e unione dataset...")
x_raw_train = fix_emnist_axes(x_raw_train)
x_raw_test = fix_emnist_axes(x_raw_test)

x_all = np.concatenate((x_raw_train, x_raw_test), axis=0)
y_all = np.concatenate((y_raw_train, y_raw_test), axis=0)

def filter_and_remap(x, y, target_indices):
    mask = np.isin(y, target_indices)
    x_filtered = x[mask]
    y_filtered = y[mask]

    y_remapped = np.zeros_like(y_filtered)
    for i, orig_idx in enumerate(target_indices):
        y_remapped[y_filtered == orig_idx] = i
        count = np.sum(y_filtered == orig_idx)
        print(f"Lettera {classi_volute[i]} (orig {orig_idx}): {count} campioni trovati.")

    return x_filtered, y_remapped

x_filtered, y_filtered = filter_and_remap(x_all, y_all, indici_voluti)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    x_filtered, y_filtered, test_size=0.2, random_state=42, stratify=y_filtered
)

# Normalizzazione e Reshape
X_train = X_train_raw.astype("float32") / 255.0
X_test = X_test_raw.astype("float32") / 255.0

# Se le immagini non sono 28x28 (ma 32x32), facciamo il resize
if X_train.shape[1] != 28:
    X_train = tf.image.resize(X_train, [28, 28]).numpy()
    X_test = tf.image.resize(X_test, [28, 28]).numpy()

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

print(f"\nTraining set: {X_train.shape}, Test set: {X_test.shape}")

plt.figure(figsize=(10, 3))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(X_train[i].reshape(28, 28), cmap='gray')
    plt.title(f"Label: {classi_volute[y_train[i]]}")
    plt.axis('off')
plt.show()

# 5. Definizione Modello
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(classi_volute), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=15,         # più rotazione
    width_shift_range=0.20,    # più spostamento (simula decentramento)
    height_shift_range=0.20,
    zoom_range=0.15,           # più variazione di scala
    shear_range=0.10,          # ← NUOVO: deforma leggermente la forma
    fill_mode='constant',
    cval=0                     # ← fix sfondo nero
)

datagen.fit(X_train)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# 6. Addestramento
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=20,
    validation_data=(X_test, y_test), # La validazione rimane sulle immagini reali
    verbose=2,
    callbacks=[early_stop]
)

# 7. Salvataggio nella cartella 'models'
folder_name = "models"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

model_path = os.path.join(folder_name, "model.h5")
model.save(model_path)
print(f"\nModello salvato con successo in {model_path}!")

# Creazione Plot
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Matrice di confusione
y_pred = np.argmax(model.predict(X_test), axis=1)

cm = confusion_matrix(y_test, y_pred, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classi_volute)
fig, ax = plt.subplots(figsize=(7,7))
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.title("Confusion Matrix")
plt.show()

print("\nReport di classificazione:")
print(classification_report(y_test, y_pred, target_names=classi_volute))
