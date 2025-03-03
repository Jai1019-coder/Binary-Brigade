import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces_in_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found. Check the path!")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    resized_image = cv2.resize(image, (500, 500))
    cv2.imshow("Face Detection - Image", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_faces_in_webcam():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not access webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow('Webcam Face Detection (Press SPACE to Capture, Q to Quit)', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            cv2.imwrite("captured_face.jpg", frame)
            print("Image captured and saved as 'captured_face.jpg'")
            break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
