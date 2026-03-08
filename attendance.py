import cv2, csv, os
from datetime import datetime

# Load student names
students = {}
if os.path.exists("students.txt"):
    with open("students.txt", "r", encoding="utf-8") as f:
        for line in f:
            sid, name = line.strip().split(",", 1)
            students[int(sid)] = name

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer/trainer.yml")

cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cam = cv2.VideoCapture(0)
marked = set()

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray,1.2,5)

    for (x,y,w,h) in faces:
        sid, conf = recognizer.predict(gray[y:y+h, x:x+w])
        if conf < 70:
            marked.add(sid)
            name = students.get(sid, "Unknown")
            cv2.putText(img,f"{sid} - {name}",(x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)
        else:
            cv2.putText(img,"Unknown",(x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,255),2)

        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    cv2.imshow("Attendance System", img)
    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()

file_exists = os.path.isfile("Attendance.csv")

with open("Attendance.csv","a",newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["Student ID", "Student Name", "Date", "Time"])

    for sid in marked:
        now = datetime.now()
        writer.writerow([
            sid,
            students.get(sid, "Unknown"),
            now.strftime("%d-%m-%Y"),
            now.strftime("%H:%M:%S")
        ])

print("Attendance saved successfully")
