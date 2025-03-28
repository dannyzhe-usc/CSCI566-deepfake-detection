import cv2
import os

real_images_dir = 'Celeb-DF-v2/still_frames_input/real_input'
fake_images_dir = 'Celeb-DF-v2/still_frames_input/fake_input'

# destination directories for cropped images
cropped_real_images_dir = 'Celeb-DF-v2/still_frames_input/real_input_cropped'
cropped_fake_images_dir = 'Celeb-DF-v2/still_frames_input/fake_input_cropped'

os.makedirs(cropped_real_images_dir, exist_ok=True)
os.makedirs(cropped_fake_images_dir, exist_ok=True)

# OpenCV's pre-trained Haar Cascade for face detection
# works decently well but still noticeable amount of images it can't detect face when there actually is a face 
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

margin_factor = 0.2

def detect_and_crop_face(image_path, output_folder, label):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) > 0:
        (x, y, w, h) = faces[0]  

        margin_x = int(margin_factor * w)
        margin_y = int(margin_factor * h)

        x = max(x - margin_x, 0)
        y = max(y - margin_y, 0)
        w = min(w + 2 * margin_x, img.shape[1] - x)
        h = min(h + 2 * margin_y, img.shape[0] - y)

        cropped_face = img[y:y+h, x:x+w]
        cropped_face_resized = cv2.resize(cropped_face, (224, 224)) # 224 for resnet. could be diff for other cnn architectures
        
        output_filename = os.path.join(output_folder, f'cropped_{os.path.basename(image_path)}')
        cv2.imwrite(output_filename, cropped_face_resized)
        print(f"Saved cropped face for {label}: {output_filename}")
    else:
        # idk im just keeping the original image if no face detected 
        output_filename = os.path.join(output_folder, f'notcropped_{os.path.basename(image_path)}')
        cv2.imwrite(output_filename, img)
        print(f"No face detected, saved original image: {output_filename}")




for image_file in os.listdir(real_images_dir):
    image_path = os.path.join(real_images_dir, image_file)
    if os.path.isfile(image_path):
        detect_and_crop_face(image_path, cropped_real_images_dir, 'real')

for image_file in os.listdir(fake_images_dir):
    image_path = os.path.join(fake_images_dir, image_file)
    if os.path.isfile(image_path):
        detect_and_crop_face(image_path, cropped_fake_images_dir, 'fake')

print("face detection and cropping completed.")
