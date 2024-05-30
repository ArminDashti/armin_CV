import cv2
import os


def frames_to_video(imgs_dir, video_dir, video_name, fps=10):
    images = [img for img in os.listdir(imgs_dir) if img.endswith(".jpg")]
    images.sort()
    frame = cv2.imread(os.path.join(imgs_dir, images[0]))
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video = cv2.VideoWriter(output_dir+'/'+video_name, fourcc, fps, (width, height))
    for image in images:
        video.write(cv2.imread(os.path.join(imgs_dir, image)))


def frames_to_video(frames_dir, video_dir, video_name, fps=10, video_format='mp4'):
    width, height = 300, 300
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_dir = os.path.join(video_dir, video_name)
    video = cv2.VideoWriter(video_dir, fourcc, 30, (width, height))
#%%
import os
os.path.join('\\home\', 'armin', 'face')