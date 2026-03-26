import cv2
import numpy as np

def get_classified_grid(image_path, model, classes):
    image = cv2.imread(image_path)
    original = image.copy() 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    blur = cv2.GaussianBlur(gray, (5, 5), 0) 
    
    # Threshold adattivo per isolare la griglia
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2) 
    
    # Trova i contorni delle celle
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    
    detected_cells = []
    
    for c in contours:
        area = cv2.contourArea(c) 
        if area > 2000: # Filtro dimensione cella
            x, y, w, h = cv2.boundingRect(c) 
            
            # Estrazione ROI in memoria (niente salvataggio su disco) 
            roi = original[y:y+h, x:x+w]
            
            # Pre-processing per il modello (Resize a 28x28 e Normalizzazione)
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi_resized = cv2.resize(roi_gray, (28, 28)).astype("float32") / 255.0
            roi_reshaped = roi_resized.reshape(1, 28, 28, 1)
            
            # Predizione immediata
            prediction = model.predict(roi_reshaped)
            label_idx = np.argmax(prediction)
            label_text = classes[label_idx]
             
            detected_cells.append({
                'pos': (x, y),
                'label': label_text
            })
            
            # Feedback visivo sull'immagine
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(image, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return detected_cells, image