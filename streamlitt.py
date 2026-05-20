import streamlit as st
import numpy as np
import os
from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt

st.set_page_config(page_title="Brain Tumor K-Means", layout="wide")

dataset_path = r"C:\Users\dell\.cache\kagglehub\datasets\masoudnickparvar\brain-tumor-mri-dataset\versions\2"

classes = ["glioma", "meningioma", "notumor", "pituitary"]

K = 4



def image_to_vector(file):#Elle prend un argument file (le chemin de l’image ou un fichier image)

    img = Image.open(file).convert("L")#On ouvre l’image avec PIL (Image.open) #.convert("L") transforme l’image en niveau de gris (grayscale)
    img = img.resize((64,64))#On force toutes les images à avoir la même taille

    img = np.array(img) / 255.0#On convertit l’image en tableau NumPy :Chaque pixel devient un nombre entre 0 et 255,Puis on divise par 255 :,On normalise les valeurs entre 0 et 1

    return img.flatten()#On transforme l’image 2D (64 × 64) en vecteur 1D



@st.cache_resource
def load_data():

    X = []
    y = []

    for cls in classes:

        folder = os.path.join(dataset_path, "Training", cls)

        images = os.listdir(folder)[:]

        for img_name in images:

            path = os.path.join(folder, img_name)

            X.append(image_to_vector(path))
            y.append(cls)

    return np.array(X), y








def kmeans(X, K=4, tol=1e-4, max_iter=100):

    centroids = X[np.random.choice(len(X), K, replace=False)]

    prev_wcss = float("inf")

    for _ in range(max_iter):

        clusters = [[] for _ in range(K)]

        wcss = 0

        # Attribution des points
        for x in X:
            dists = [np.linalg.norm(x - c) for c in centroids]
            idx = np.argmin(dists)
            clusters[idx].append(x)

            # contribution au WCSS
            wcss += np.linalg.norm(x - centroids[idx])**2

        # Mise à jour des centroïdes
        new_centroids = []

        for i in range(K):
            if len(clusters[i]) == 0:
                new_centroids.append(centroids[i])
            else:
                new_centroids.append(np.mean(clusters[i], axis=0))

        centroids = np.array(new_centroids)

        # condition d'arrêt (stabilisation du WCSS)
        if abs(prev_wcss - wcss) < tol:
            break

        prev_wcss = wcss

    return centroids








@st.cache_resource
def train_model():

    X, y = load_data()

    centroids = kmeans(X)

    labels = []

    for x in X:

        dists = [np.linalg.norm(x - c) for c in centroids]

        labels.append(np.argmin(dists))

    labels = np.array(labels)

    cluster_to_class = {}

    for k in range(K):

        idxs = np.where(labels == k)[0]

        true_labels = [y[i] for i in idxs]

        cluster_to_class[k] = Counter(true_labels).most_common(1)[0][0]

    return centroids, cluster_to_class, X, labels









def predict(file, centroids, cluster_to_class):

    x = image_to_vector(file)

    dists = [np.linalg.norm(x - c) for c in centroids]

    cluster = np.argmin(dists)

    return cluster_to_class[cluster]



















st.title("🧠 Brain Tumor Detection -- K-Means PRO")

centroids, cluster_to_class, X, labels = train_model()



# UPLOAD
uploaded = st.file_uploader("Upload MRI Image", type=["jpg","png","jpeg"])

if uploaded:

    st.image(uploaded, width=250)

    prediction = predict(uploaded, centroids, cluster_to_class)

    st.success(f"Prediction: {prediction}")
