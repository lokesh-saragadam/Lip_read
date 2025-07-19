# Import all of the dependencies
import streamlit as st
import os 
import imageio 
import ffmpeg
import tensorflow as tf 
from utils import load_data, num_to_char
from modelutils import load_model
import numpy as np
st.set_page_config(layout='wide')


# Setup the sidebar
with st.sidebar: 
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('LipBuddy')
    st.info('This application is originally developed from the LipNet deep learning model.')

st.title('LipNet Full Stack App')
#set the layout to the streamlit...
options = os.listdir(os.path.join('..','data','s1'))
selected_video = st.selectbox('Choose video', options)

#generate two columns
col1,col2 = st.columns(2)

if options:

    with col1:
        st.info('The video below displayes the converted video in mp4 format')
        file_path = os.path.join('..','data','s1',selected_video)
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')
        
        #Rendering inside of the app
        video = open('test_video.mp4','rb')
        video_bytes = video.read()
        st.video(video_bytes)

    with col2:
        st.info("This is all the machine learning model sees when predicting")
        video, annotation = load_data(tf.convert_to_tensor(file_path))
        # Convert to NumPy if needed
        if isinstance(video, tf.Tensor):
            video = video.numpy()  # shape: (N, 46, 140, 1)

        # Remove extra grayscale channel
        if video.shape[-1] == 1:
            video = np.squeeze(video, axis=-1)  # shape: (N, 46, 140)

        video = (video * 255).clip(0, 255).astype(np.uint8)
        
        # Save GIF
        imageio.mimsave("animation.gif", video, fps=10)
        st.image('animation.gif',width=400)

        st.info('This is the output of the machine learning model as tokens')
        model = load_model()
        yhat = model.predict(tf.expand_dims(video,axis=0))
        
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy = True)[0][0].numpy()
        st.text(decoder)
        st.info('Decode the raw tokens into words')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')

        st.text(converted_prediction)
        #Num to Char   