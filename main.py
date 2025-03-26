import os

print("Fire Detection System")
print("1. Train the Model")
print("2. Run Fire Detection")
choice = input("Enter choice: ")

if choice == "1":
    os.system("python3 Scripts/train.py")
elif choice == "2":
    os.system("python3 Scripts/detect.py")
else:
    print("Invalid choice")

