import cv2
import os

# for now just take only the first frame from each video. 
# later want to extract certain number of frames from each video, or 1 frame per certain time interval 

real_videos_folder = 'Celeb-DF-v2/Celeb-real'
fake_videos_folder = 'Celeb-DF-v2/Celeb-synthesis'

# Output folders for the first frames
real_frames_folder = 'Celeb-DF-v2/still_frames_input/real_input'
fake_frames_folder = 'Celeb-DF-v2/still_frames_input/fake_input'

os.makedirs(real_frames_folder, exist_ok=True)
os.makedirs(fake_frames_folder, exist_ok=True)


def extract_first_frame(video_path, output_folder, label):
    cap = cv2.VideoCapture(video_path)
    
    if cap.isOpened():
        ret, frame = cap.read()
        
        if ret:
            frame_filename = os.path.join(output_folder, f'{label}_{os.path.basename(video_path)}.jpg')
            cv2.imwrite(frame_filename, frame)
            print(f'Extracted {frame_filename}')
        else:
            print(f'Error reading frame from {video_path}')
    else:
        print(f'Failed to open video {video_path}')
    
    cap.release()




for video_file in os.listdir(real_videos_folder):
    video_path = os.path.join(real_videos_folder, video_file)
    if os.path.isfile(video_path):
        extract_first_frame(video_path, real_frames_folder, 'real')

for video_file in os.listdir(fake_videos_folder):
    video_path = os.path.join(fake_videos_folder, video_file)
    if os.path.isfile(video_path):
        extract_first_frame(video_path, fake_frames_folder, 'fake')
