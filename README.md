# Greyscale_Image_Colorization_and_Super_Resolution

This Machine Learning Model takes in Greyscale Low-Light Image as input and produces Colorized High-Resolution Image as output.

For Image Colorization, following GitHub Repository is used:
https://github.com/richzhang/colorization

For Low-Light Image Enhancement, Adaptive Histogram Equalization is used.

For Image Super-Resolution, following GitHub Repository is used:
https://github.com/ai-forever/Real-ESRGAN

For using this model, proceed as per following steps:

```python
!git clone https://github.com/richzhang/colorization.git
!git clone https://github.com/sberbank-ai/Real-ESRGAN.git
```
```python
streamlit run app.py
```
