import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageOps
import numpy as np

st.set_option('deprecation.showfileUploaderEncoding', False)

def teachable_machine_classification(img, file):
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = keras.models.load_model(file)

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = img
    # image = Image.open(img_name).convert('RGB')
    # image = cv2.imread(image)

    # resize the image to a 224x224 with the same strategy as in TM2:
    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # display the resized image
    image.show()

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    #print(prediction)
    return np.argmax(prediction)


st.title("Clasificación de Rayos X con Machine Learning")
st.header("Normal vs. Virus")
st.write("Carga tu radiografía para identificar si se detecta la presencia de un **virus**")
# file upload and handling logic
uploaded_file = st.file_uploader("Carga una imagen en formato jpeg", type="jpeg")
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
#image = Image.open(img_name).convert('RGB')
    st.image(image, caption='Imagen cargada.', use_column_width=True)
    st.write("")

    import time
    latest_iteration = st.empty()
    bar = st.progress(0)

    for i in range(100):
    # Update the progress bar with each iteration.
        latest_iteration.text(f'Procesamiento de la imagen {i+1}')
        bar.progress(i + 1)
        time.sleep(0.1)

    label = teachable_machine_classification(image, r'keras_model.h5')
    print (label)
    if label == 1:
        st.write("Esta radiografía pulmonar presenta opacidades anormales y requiere investigación a detalle por parte de un especialista.")
    else:
        st.write("Esta radiografía pulmonar parece normal y no muestra áreas de opacidad anormales")
