import cv2
import os

# Paths to the real and fake image directories
real_images_dir = 'Celeb-DF-v2/still_frames_input/real_input'
fake_images_dir = 'Celeb-DF-v2/still_frames_input/fake_input'

# Paths to save cropped faces
cropped_real_images_dir = 'Celeb-DF-v2/still_frames_input/real_input_cropped'
cropped_fake_images_dir = 'Celeb-DF-v2/still_frames_input/fake_input_cropped'

# Create directories if they don't exist
os.makedirs(cropped_real_images_dir, exist_ok=True)
os.makedirs(cropped_fake_images_dir, exist_ok=True)

# Load OpenCV's pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Margin factor (how much to expand the bounding box around the face)
margin_factor = 0.2  # For example, 20% margin

# Function to detect and crop faces from images
def detect_and_crop_face(image_path, output_folder, label):
    # Read the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for face detection
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # If faces are detected, crop the first one and add a margin
    if len(faces) > 0:
        (x, y, w, h) = faces[0]  # Take the first detected face
        
        # Add margin around the face
        margin_x = int(margin_factor * w)
        margin_y = int(margin_factor * h)
        
        # Expand the bounding box with the margin, but make sure it stays within image boundaries
        x = max(x - margin_x, 0)
        y = max(y - margin_y, 0)
        w = min(w + 2 * margin_x, img.shape[1] - x)
        h = min(h + 2 * margin_y, img.shape[0] - y)
        
        # Crop the face with the new coordinates
        cropped_face = img[y:y+h, x:x+w]
        
        # Resize the cropped face to match the input size of the model (224x224 for ResNet50)
        cropped_face_resized = cv2.resize(cropped_face, (224, 224))
        
        # Save the cropped face
        output_filename = os.path.join(output_folder, f'cropped_{os.path.basename(image_path)}')
        cv2.imwrite(output_filename, cropped_face_resized)
        print(f"Saved cropped face for {label}: {output_filename}")
    else:
        # If no face is detected, save the original image as is
        output_filename = os.path.join(output_folder, f'notcropped_{os.path.basename(image_path)}')
        cv2.imwrite(output_filename, img)
        print(f"No face detected, saved original image: {output_filename}")

# Detect and crop faces in the real images
for image_file in os.listdir(real_images_dir):
    image_path = os.path.join(real_images_dir, image_file)
    if os.path.isfile(image_path):
        detect_and_crop_face(image_path, cropped_real_images_dir, 'real')

# Detect and crop faces in the fake images
for image_file in os.listdir(fake_images_dir):
    image_path = os.path.join(fake_images_dir, image_file)
    if os.path.isfile(image_path):
        detect_and_crop_face(image_path, cropped_fake_images_dir, 'fake')

print("Face detection and cropping completed.")
