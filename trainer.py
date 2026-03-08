
import cv2, os
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def getImages(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faces, ids = [], []

    for imagePath in imagePaths:
        img = Image.open(imagePath).convert('L')
        img_np = np.array(img,'uint8')
        id = int(imagePath.split('.')[1])
        detected = detector.detectMultiScale(img_np)

        for (x,y,w,h) in detected:
            faces.append(img_np[y:y+h, x:x+w])
            ids.append(id)

    return faces, ids

faces, ids = getImages("dataset")
recognizer.train(faces, np.array(ids))

os.makedirs("trainer", exist_ok=True)
recognizer.save("trainer/trainer.yml")
print("Model trained successfully")
