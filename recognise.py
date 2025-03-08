import numpy as np
import pickle
import os
import cv2
import time
import imutils
from collections import defaultdict

curr_path = os.getcwd()

print("Loading face detection model")
proto_path = os.path.join(curr_path, 'model', 'deploy.prototxt')
model_path = os.path.join(curr_path, 'model', 'res10_300x300_ssd_iter_140000.caffemodel')
face_detector = cv2.dnn.readNetFromCaffe(prototxt=proto_path, caffeModel=model_path)

print("Loading face recognition model")
recognition_model = os.path.join(curr_path, 'model', 'openface_nn4.small2.v1.t7')
face_recognizer = cv2.dnn.readNetFromTorch(model=recognition_model)

print("Loading trained recognizer and label encoder")
recognizer = pickle.loads(open('recognizer.pickle', "rb").read())
le = pickle.loads(open('le.pickle', "rb").read())

print("Labels loaded by label encoder:")
print(le.classes_)

print("Starting test video file")
vs = cv2.VideoCapture(r"C:\Users\prash\Desktop\prash project\VID-20240218-WA0031.mp4") #r"C:\Users\prash\Desktop\prash project\VID-20240218-WA0031.mp4"
time.sleep(1)

# Dictionary to store profile images
profiles = {}
profiles_path = os.path.join(curr_path, 'profiles')
profile_image_size = (100, 100)  # Define a standard size for profile images

# Dictionary to count recognitions
recognition_counts = {label: 0 for label in le.classes_}

# Dictionary to track last detection time
last_detected = defaultdict(lambda: time.time())

# Load profile images
print("Loading profile images")
for label in le.classes_:
    profile_img_path = os.path.join(profiles_path, f"{label}.jpg")
    print(f"Checking for profile image at: {profile_img_path}")
    if os.path.exists(profile_img_path):
        img = cv2.imread(profile_img_path)
        profiles[label] = cv2.resize(img, profile_image_size)
        print(f"Loaded profile image for {label}")
    else:
        print(f"No profile image found for {label}")

while True:
    ret, frame = vs.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = imutils.resize(frame, width=1000, height=500)
    (h, w) = frame.shape[:2]
    image_blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)

    face_detector.setInput(image_blob)
    face_detections = face_detector.forward()

    current_time = time.time()
    detection_interval = 1.0  # Minimum time (in seconds) between detections to count

    for i in range(0, face_detections.shape[2]):
        confidence = face_detections[0, 0, i, 2]

        if confidence >= 0.5:
            box = face_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = frame[startY:endY, startX:endX]

            (fH, fW) = face.shape[:2]

            if fW < 20 or fH < 20:
                continue

            face_blob = cv2.dnn.blobFromImage(face, 1.0/255, (96, 96), (0, 0, 0), True, False)

            face_recognizer.setInput(face_blob)
            vec = face_recognizer.forward()

            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]

            text = "{}: {:.2f}".format(name, proba * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

            # Update recognition count if enough time has passed
            if current_time - last_detected[name] >= detection_interval:
                recognition_counts[name] += 1
                last_detected[name] = current_time

            if name in profiles:
                profile_img = profiles[name]
                img_h, img_w = profile_img.shape[:2]

                # Position the profile image to the right of the bounding box
                overlay_x = endX + 10 if endX + 10 + img_w < w else startX - img_w - 10
                overlay_y = startY

                # Check if the profile image fits inside the frame
                if overlay_y + img_h < h and overlay_x + img_w < w:
                    frame[overlay_y:overlay_y + img_h, overlay_x:overlay_x + img_w] = profile_img

                    # Display the recognition count below the profile image
                    count_text = f"Count: {recognition_counts[name]}"
                    count_y = overlay_y + img_h + 20 if overlay_y + img_h + 20 < h else overlay_y + img_h - 10
                    cv2.putText(frame, count_text, (overlay_x, count_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    cv2.imshow("Video Stream", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

cv2.destroyAllWindows()
vs.release()
