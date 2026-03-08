
import tkinter as tk
import os

def capture():
    os.system("python dataset_creator.py")

def train():
    os.system("python trainer.py")

def attendance():
    os.system("python attendance.py")

root = tk.Tk()
root.title("Smart Attendance System")
root.geometry("400x300")

tk.Label(root, text="Smart Attendance System", font=("Arial", 18, "bold")).pack(pady=20)

tk.Button(root, text="Capture Face Data", width=25, command=capture).pack(pady=10)
tk.Button(root, text="Train Model", width=25, command=train).pack(pady=10)
tk.Button(root, text="Mark Attendance", width=25, command=attendance).pack(pady=10)

tk.Label(root, text="Press ESC to close camera").pack(pady=10)
root.mainloop()
