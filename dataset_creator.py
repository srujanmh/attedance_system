import cv2
import os

cam = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

student_id = input("Enter Student ID: ")
student_name = input("Enter Student Name: ")

# Save ID–Name mapping
with open("students.txt", "a", encoding="utf-8") as f:
    f.write(f"{student_id},{student_name}\n")

os.makedirs("dataset", exist_ok=True)
count = 0

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        count += 1
        cv2.imwrite(
            f"dataset/User.{student_id}.{count}.jpg",
            gray[y:y+h, x:x+w]
        )
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    cv2.imshow("Capture Face Data", img)

    if cv2.waitKey(1) == 27 or count >= 50:
        break

cam.release()
cv2.destroyAllWindows()
print("Face data collected successfully")
