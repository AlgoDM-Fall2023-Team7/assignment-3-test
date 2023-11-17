import streamlit as st
import tensorflow as tf
import numpy as np
import ntpath
import cv2
import glob
from sklearn import manifold
import matplotlib.pyplot as plt
import matplotlib.offsetbox as offsetbox
from scipy.spatial.distance import cdist

# Set a custom title for your Streamlit app using Markdown
st.markdown("<h1 style='color: blue; font-weight: bold;'>Image Search By an Artistic Style</h1>", unsafe_allow_html=True)

# Load the TensorFlow model
model = tf.keras.models.load_model(r"assignment-3-test/artistic_model.keras")

def style_to_vec(style):
    # Flatten and convert the style tensor to a NumPy array
    return style.numpy().ravel()

image_paths = glob.glob(r'assignment-3-test/images-by-style/*.jpg')

def load_image(image):
    image = plt.imread(image)
    img = tf.image.convert_image_dtype(image, tf.float32)
    img = tf.image.resize(img, [400, 400])
    img = img[tf.newaxis, :]  # Shape -> (batch_size, h, w, d)
    return img

@tf.function
def image_to_style(image_tensor, model):
    style = model(image_tensor)
    return style

image_style_embeddings = {}
for image_path in image_paths:
    image_tensor = load_image(image_path)
    style = style_to_vec(image_to_style(image_tensor, model))
    image_style_embeddings[ntpath.basename(image_path)] = style

images = {}
for image_path in image_paths:
    image = cv2.imread(image_path, 3)
    b, g, r = cv2.split(image)  # Get b, g, r
    image = cv2.merge([r, g, b])  # Switch it to r, g, b
    image = cv2.resize(image, (200, 200))
    images[ntpath.basename(image_path)] = image

# Calculate similarity between embeddings (you can choose another distance metric)
embeddings_matrix = np.array(list(image_style_embeddings.values()))
distances = cdist(embeddings_matrix, embeddings_matrix, 'euclidean')

# Allow the user to upload an image for similarity search
query_image = st.file_uploader("Upload a Query Image", type=["jpg", "jpeg", "png"])

# Check if an image has been uploaded
if query_image is not None:
    # Display the selected image
    st.subheader("Selected Image:")
    st.image(query_image, caption="Uploaded Image", use_column_width=True)

    # Load and preprocess the uploaded image
    query_image = tf.image.decode_image(query_image.read(), channels=3)
    query_image = tf.image.convert_image_dtype(query_image, tf.float32)
    query_image = tf.image.resize(query_image, [400, 400])
    query_image = query_image[tf.newaxis, :]  # Shape -> (batch_size, h, w, d)

    # Calculate the style embedding for the uploaded image
    query_style = style_to_vec(image_to_style(query_image, model))

    # Calculate similarity between the uploaded image and pre-loaded embeddings
    similarities = np.linalg.norm(embeddings_matrix - query_style, axis=1)

    # Get and display the top 5 similar images to the uploaded image
    similar_indices = np.argsort(similarities)[:5]
    similar_images = [list(image_style_embeddings.keys())[index] for index in similar_indices]

    st.subheader("Top 5 Similar Images:")
    for similar_image_name in similar_images:
        st.image(images[similar_image_name], caption=similar_image_name, use_column_width=True)
