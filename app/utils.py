import tensorflow as tf
from typing import List
import cv2
import os 


vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
# Mapping integers back to original characters
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)


def load_video(path: str) -> tf.Tensor:
    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))): 
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = tf.image.rgb_to_grayscale(frame)  
        frame = frame[190:236, 80:220, :]  # crop
        frames.append(frame)
    cap.release()

    frames = tf.stack(frames)  # ✅ (N, H, W, 1)
    frames = tf.cast(frames, tf.float32)
    mean = tf.reduce_mean(frames)
    std = tf.math.reduce_std(frames)
    frames = (frames - mean) / std
    frames = (frames - tf.reduce_min(frames)) / (tf.reduce_max(frames) - tf.reduce_min(frames))  # normalize to [0, 1]
    return frames
    

def load_alignments(path:str) -> List[str]: 
    #print(path)
    with open(path, 'r') as f: 
        lines = f.readlines() 
    tokens = []
    for line in lines:
        line = line.split()
        if line[2] != 'sil': 
            tokens = [*tokens,' ',line[2]]
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]

def load_data(path: str): 
    path = bytes.decode(path.numpy())
    file_name = path.split('/')[-1].split('.')[0]
    # File name splitting for windows
    # file_name = path.split('\\')[-1].split('.')[0]
    video_path = os.path.join('data','s1',f'{file_name}.mpg')
    alignment_path = os.path.join('data','alignments','s1',f'{file_name}.align')
    frames = load_video(video_path) 
    alignments = load_alignments(alignment_path)
    
    return frames, alignments