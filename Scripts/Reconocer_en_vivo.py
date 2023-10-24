import time
from keras.models import load_model
import cv2
import numpy as np

modelo1 = load_model('InceptionV3_RGB/')
modelo2 = load_model('InceptionV3_GV3/')
modelo3 = load_model('model_MobileNetV2/')

while True:
    
    m = input("\n\n- 1: InceptionV3_RGB\n- 2: InceptionV3_GV3\n- 3: model_MobileNetV2 \n")

    if (m == "1"):
        modelo = modelo1
        name = "InceptionV3_RGB"
        print("Modelo: InceptionV3_RGB")
        time.sleep(2)
    elif (m == "2"):
        modelo = modelo2
        name = "InceptionV3_GV3"
        print("Modelo: InceptionV3_GV3")
        time.sleep(2)
    else:
        modelo = modelo3
        name = "model_MobileNetV2"
        print("Modelo: model_MobileNetV2")
        time.sleep(2)


    face_clsfr = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    labels_dict = {0: 'Facundo', 1: 'Luisma', 2: 'Martin', 3: 'Matias'}
    color_dict = {0: (255, 178, 0), 1: (0, 0, 255), 2: (0, 255, 0), 3: (255, 255, 255)}


    source = cv2.VideoCapture(0)

    while True:
        ret, img = source.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_clsfr.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        for (x, y, w, h) in faces:
        
            # face_img = img[y:y + h, x:x + w]
            
            face_img = gray[y:y + h, x:x + w]
            face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2BGR)
            

            if not face_img.size == 0:
                face_img = cv2.resize(face_img, (224, 224))
                norm_face = face_img / 255.0
                
                probability = modelo.predict(np.expand_dims(norm_face, axis=0))[0]
                probability = np.around(probability*100, decimals=1)
                
                class_num = np.argmax(probability)
                
                class_probabilities = [f"{labels_dict[i]}: {probability[i]:.2f}" for i in range(len(labels_dict))]
                print(", ".join(class_probabilities))
                
                class_predicted = labels_dict[class_num]
                print(class_predicted,"\n")
                
                cv2.putText(img, f"{class_predicted}: {max(probability):.2f}%", (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color_dict[class_num], 2)
                cv2.rectangle(img, (x, y), (x + w, y + h), color_dict[class_num], 2)
                
        cv2.imshow(name, img)
        key = cv2.waitKey(1)

        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    source.release()