import os
import cv2

import os
from PIL import Image

import os
import cv2
from PIL import Image

def extract_frames(root_dir, target_folder, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for subdir, _, files in os.walk(root_dir):
        if target_folder not in subdir.split(os.sep):
            continue

        relative_path = os.path.relpath(subdir, root_dir)

        for file in files:
            file_path = os.path.join(subdir, file)
            file_name, ext = os.path.splitext(file)
            frame_output_dir = os.path.join(output_dir, relative_path, file_name)
            os.makedirs(frame_output_dir, exist_ok=True)

            try:
                if ext.lower() == '.gif':
                    gif = Image.open(file_path)
                    frame_idx = 0

                    while True:
                        frame = gif.convert('RGB')
                        frame.save(os.path.join(frame_output_dir, f'frame_{frame_idx:04d}.jpg'), 'JPEG')
                        frame_idx += 1
                        gif.seek(gif.tell() + 1)
                elif ext.lower() == '.mp4':
                    cap = cv2.VideoCapture(file_path)
                    frame_idx = 0

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame_path = os.path.join(frame_output_dir, f'frame_{frame_idx:04d}.jpg')
                        cv2.imwrite(frame_path, frame)
                        frame_idx += 1

                    cap.release()
                else:
                    print(f"Skipped unsupported file type: {file_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")


# Example usage
#extract_frames_from_gifs(
#    root_dir='outputs_v11/flatten-results_cfg_scale_7.5/explicit/interventions',
#    target_folder='age',  # Replace with 'gender', 'beard', or 'bald' as needed
#    output_dir='output_gif_frames'
#)


# Example usage
for attr in ["age", "gender", "beard", "bald"]:
    extract_frames(
        root_dir='outputs_rephrasing_LLM_v2/flatten-results_cfg_scale_7.5/explicit/interventions',
        target_folder=attr,  # Change to 'gender', 'beard', or 'bald' as needed
        output_dir='output_frames_flatten_rephrasing_LLM_v2',
        #frame_rate=1  # Extract one frame per second from 20 fps video/storage/flatten-results_cfg_scale_7.5
    )
