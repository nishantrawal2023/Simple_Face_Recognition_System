import cv2
import os
import numpy as np
from PIL import Image
import streamlit as st

# Paths
IMAGES_DIR = "images"
TRAINER_FILE = "trainer.yml"
CASCADE_FILE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Ensure `images` folder exists
os.makedirs(IMAGES_DIR, exist_ok=True)

# Initialize face recognizer and cascade
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(CASCADE_FILE)

# Face Taker
def capture_faces(name, id):
    count = 0
    cap = cv2.VideoCapture(0)
    st.info("Press 'Q' to stop capturing images.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        for (x, y, w, h) in faces:
            count += 1
            file_path = os.path.join(IMAGES_DIR, f"{name}-{id}-{count}.jpg")
            cv2.imwrite(file_path, gray[y:y+h, x:x+w])
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow("Capturing Faces", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    st.success(f"{count} images captured and saved in the 'images' folder.")


# Trainer
def train_recognizer():
    def get_images_and_labels(image_dir):
        image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]
        face_samples = []
        ids = []
        for image_path in image_paths:
            try:
                if not image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.pgm')):
                    raise ValueError("Invalid image format.")
                gray_image = Image.open(image_path).convert("L")
                image_np = np.array(gray_image, "uint8")
                filename = os.path.basename(image_path)
                id = int(filename.split("-")[1])  # Extract ID from filename
                faces = detector.detectMultiScale(image_np)
                for (x, y, w, h) in faces:
                    face_samples.append(image_np[y:y+h, x:x+w])
                    ids.append(id)
            except Exception as e:
                st.warning(f"Skipping {image_path}: {e}")
                continue
        return face_samples, ids

    st.info("Training the face recognition model...")
    faces, ids = get_images_and_labels(IMAGES_DIR)
    if faces and ids:
        recognizer.train(faces, np.array(ids))
        recognizer.write(TRAINER_FILE)
        st.success(f"Training complete. {len(set(ids))} unique faces trained.")
    else:
        st.error("No valid images found for training.")


# Recognizer
def recognize_faces():
    recognizer.read(TRAINER_FILE)
    cap = cv2.VideoCapture(0)
    st.info("Press 'Q' to exit recognition mode.")

    recognized_faces = set()  # Store unique recognized IDs

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        # Track recognized faces
        for (x, y, w, h) in faces:
            id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

            if confidence < 100:  # Only recognize faces with high confidence
                name_id = f"ID: {id}"
                recognized_faces.add(id)  # Add to recognized faces set
                confidence_text = f"{round(100 - confidence)}% confident"
            else:
                name_id = "Unknown"
                confidence_text = ""
                recognized_faces.discard(id)  # Remove unknown faces

            cv2.putText(frame, name_id, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.putText(frame, confidence_text, (x+5, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Show the number of unique recognized faces in real-time
        recognized_count = len(recognized_faces)
        cv2.putText(frame, f"Recognized Faces: {recognized_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        # Display the live camera feed
        cv2.imshow("Recognizing Faces", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# Streamlit App
st.title("Face Recognition System")

# Face Taker Section
st.header("Face Taker")
name = st.text_input("Enter Name")
id = st.text_input("Enter ID")
if st.button("Capture Faces"):
    if name and id.isdigit():
        capture_faces(name, int(id))
    else:
        st.error("Please enter a valid name and numeric ID.")

# Trainer Section
st.header("Trainer")
if st.button("Train Model"):
    train_recognizer()

# Recognizer Section
st.header("Recognizer")
if st.button("Start Recognition"):
    recognize_faces()
