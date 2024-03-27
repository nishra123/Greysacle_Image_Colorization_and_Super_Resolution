import os
import time
import streamlit as st
from skimage import io, color, exposure, img_as_ubyte
import matplotlib.pyplot as plt
import torch
from PIL import Image
import numpy as np

# Change the current working directory
new_directory = r'C:\Users\nipan\OneDrive\Desktop\IMAGE COLORIZATION ML'  # Specify your desired directory path here
os.chdir(new_directory)

from RealESRGAN import RealESRGAN

def main():
    st.title("Grayscale Image Colorization and Super-Resolution")

    # File uploader for image input
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

        if st.button("Start Processing"):
            # Perform operations on the uploaded image
            start_processing_time = time.time()

            # Save uploaded image
            with open('uploaded_image.jpg', 'wb') as f:
                f.write(uploaded_file.getvalue())

            # Read the uploaded image
            im1 = io.imread("uploaded_image.jpg")
            im1 = color.rgb2gray(im1)
            st.image(im1, caption='Original Grayscale Image', use_column_width=True)

            # Perform adaptive histogram equalization (AHE) for local brightness enhancement
            im1_enhanced = exposure.equalize_adapthist(im1, clip_limit=0.03)

            # Convert the enhanced image to unsigned byte format
            im1_enhanced = img_as_ubyte(im1_enhanced)
            st.image(im1_enhanced, caption='Enhanced Grayscale Image', use_column_width=True)

            # Save the enhanced image
            io.imsave('enhanced_image.png', im1_enhanced)

            # Colorization
            os.system('python colorization/demo_release.py -i "enhanced_image.png"')

            # Super-resolution
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = RealESRGAN(device, scale=4)
            model.load_weights('weights/RealESRGAN_x4.pth', download=True)
            path_to_image = 'saved_eccv16.png'
            image = Image.open(path_to_image).convert('RGB')
            st.image(image, caption='Colored Grayscale Image', use_column_width=True)
            sr_image = model.predict(image)

            # Display time taken for processing
            processing_time = time.time() - start_processing_time
            st.write(f"Time taken for processing: {processing_time:.2f} seconds")

            # Display processed images
            
            
            
            st.image(np.array(sr_image), caption='Super-resolved Final Image', use_column_width=True)

if __name__ == "__main__":
    main()
