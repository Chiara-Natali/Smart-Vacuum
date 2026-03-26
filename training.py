import numpy as np
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from extra_keras_datasets import emnist # Libreria per il download automatico

classes = ['V', 'D', 'C', 'X', 'S', 'F']
# Mappa EMNIST: A=1, B=2, C=3, D=4, E=5, F=6, ..., S=19, ..., V=22, X=24
emnist_map = {'C': 3, 'D': 4, 'F': 6, 'S': 19, 'V': 22, 'X': 24}
target_letters = [emnist_map[c] for c in classes]

#Caricamento automatico del dataset
print("Scaricamento e caricamento eMNIST in corso...")
(x_raw, y_raw), (x_test_raw, y_test_raw) = emnist.load_data(type='letters')

#Funzione di filtraggio per tenere solo le 6 lettere
def filter_data(x, y, target_letters):
    mask = np.isin(y, target_letters) # Maschera di selezione
    x_filtered = x[mask]
    y_filtered = y[mask]
    
    #Riassegniamo le etichette da 0 a 5 per la nostra rete
    for new_index, old_label in enumerate(target_letters):
        y_filtered[y_filtered == old_label] = new_index
    return x_filtered, y_filtered

x_filtered, y_filtered = filter_data(x_raw, y_raw, target_letters)

#Pre-processing
img_size = 28
# Normalizzazione e reshape immagini
X = x_filtered.astype("float32") / 255.0
X = X.reshape(-1, img_size, img_size, 1)

#Conversione in categorie (one-hot encoding)
y = tf.keras.utils.to_categorical(y_filtered, num_classes=len(classes))

# Suddivisione Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Definizione Modello CNN
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(classes), activation='softmax')
])

#Compilazione e Addestramento
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=7, batch_size=16, validation_data=(X_test, y_test))

#Salvataggio
folder_name= "models"
model_path = os.path.join(folder_name, "model.h5")

if not os.path.exist(folder_name):
    os.makedirs(folder_name)

model.save("model.h5")
print("Modello addestrato sulle 6 lettere e salvato in {folder_name}!")