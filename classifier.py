import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def classify_grid(model_path, image_path):
    model = tf.keras.models.load_model(model_path, compile=False)
    image = cv2.imread(image_path)
    if image is None:
        print("Errore: Immagine non trovata.")
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Pre-processing
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Contorno esterno
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("Nessun contorno trovato.")
        return []

    main_grid_cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(main_grid_cnt)

    # Divisione matematica 3x3
    cell_w = w // 3
    cell_h = h // 3

    classes = ['S', 'F', 'D', 'V', 'X', 'C']
    recognized_letters = []

    plt.figure(figsize=(8, 8))

    for i in range(3): # Righe
        for j in range(3): # Colonne
            # Calcolo coordinate della cella corrente
            cx = x + (j * cell_w)
            cy = y + (i * cell_h)

            # Ritaglio della cella con un margine interno del 15% per pulire dai bordi
            margin_x = int(cell_w * 0.15)
            margin_y = int(cell_h * 0.15)
            roi = gray[cy+margin_y : cy+cell_h-margin_y, cx+margin_x : cx+cell_w-margin_x]

            # Pre-processing per il modello (Inversione e soglia)
            _, roi_bin = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

            # Resize a 28x28 (standard EMNIST)
            roi_resized = cv2.resize(roi_bin, (28, 28), interpolation=cv2.INTER_AREA)

            # Predizione
            img_input = roi_resized.reshape(1, 28, 28, 1).astype("float32") / 255.0
            pred = model.predict(img_input, verbose=0)
            letter = classes[np.argmax(pred)]
            recognized_letters.append(letter)

            # Visualizzazione per debugging
            plt.subplot(3, 3, i * 3 + j + 1)
            plt.imshow(roi_resized, cmap='gray')
            plt.title(f"Pos {i},{j}: {letter}")
            plt.axis('off')

    plt.tight_layout()
    plt.show()

    return recognized_letters
