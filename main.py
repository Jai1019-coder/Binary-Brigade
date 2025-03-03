from face_detection import detect_faces_in_webcam, detect_faces_in_image
from predict import predict_disease 

def main():
    choice = input("Enter '1' for Image Processing, '2' for Webcam Detection: ")

    if choice == '1':
        image_path = input("Enter image path: ")
        detect_faces_in_image(image_path)
    elif choice == '2':
        detect_faces_in_webcam()
        prediction = predict_disease("captured_face.jpg")
        print(f"Predicted Condition: {prediction}")
    else:
        print("Invalid choice! Please enter '1' or '2'.")

if __name__ == "__main__":
    main()
