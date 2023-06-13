# importe a biblioteca opencv
import cv2
import tensorflow as tf
import numpy as np

# defina um objeto de captura de vídeo
video = cv2.VideoCapture(0)
model = tf.keras.models.load_model('keras_model.h5')


while(True):
      
    # Capture o vídeo quadro a quadro
    check, frame = video.read()
    img = cv2.resize(frame, (224, 224))
    test_image = np.array(img, dtype = np.float32)
    test_image = np.expand_dims(test_image, axis = 0)
    normalize_image = test_image/255.0
    prediction = model.predict(normalize_image)
    print("Previsão: " , prediction)
    # Exiba o quadro resultante
    cv2.imshow('quadro', frame)
      
    # Saia da tela com a barra de espaço
    key = cv2.waitKey(1)
    
    if key == 32:
        print("Fechando...")
        break
  
# Após o loop, libere o objeto capturado
video.release()

# Destrua todas as janelas
cv2.destroyAllWindows()