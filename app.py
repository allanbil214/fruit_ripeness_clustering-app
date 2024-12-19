import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import os
import pandas as pd

def preprocessing_gambar(gambar):
    """Resize and normalize images"""
    resized_images = [cv2.resize(img, (720, 720)) for img in gambar]
    normalized_images = [cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) for img in resized_images]
    return normalized_images

def ekstrak_fitur_warna(gambar):
    """Extract color features from image"""
    gambar_hsv = cv2.cvtColor(gambar, cv2.COLOR_BGR2HSV)
    
    # Extract multiple features
    features = [
        np.mean(gambar_hsv[:, :, 0]),  # Hue
        np.mean(gambar_hsv[:, :, 1]),  # Saturation
        np.mean(gambar_hsv[:, :, 2]),  # Value
        np.std(gambar_hsv[:, :, 0]),   # Hue standard deviation
        np.std(gambar_hsv[:, :, 1]),   # Saturation standard deviation
    ]
    return features

def perform_clustering(gambar, k=2, delta=6.5):
    """Perform clustering on images"""
    progress = st.progress(0)
    status_text = st.empty()

    # Step 1: Preprocessing
    status_text.text("Step 1: Preprocessing images...")
    gambar = preprocessing_gambar(gambar)
    progress.progress(20)

    # Step 2: Extract Features
    status_text.text("Step 2: Extracting color features...")
    fitur = [ekstrak_fitur_warna(img) for img in gambar]
    fitur_array = np.array(fitur)
    fitur_hue = fitur_array[:, 0]  # Use first feature (Hue) for filtering
    progress.progress(40)

    # Step 3: Sorting and Filtering
    status_text.text("Step 3: Sorting and filtering images...")
    sorted_indices = np.argsort(fitur_hue)
    gambar_sorted = [gambar[i] for i in sorted_indices]
    fitur_sorted = fitur_array[sorted_indices]
    fitur_hue_sorted = fitur_hue[sorted_indices]

    batas_tengah = (np.max(fitur_hue_sorted) + np.min(fitur_hue_sorted)) / 2
    filter_indices = [i for i, hue in enumerate(fitur_hue_sorted) if not (batas_tengah - delta <= hue <= batas_tengah + delta)]

    gambar_cleaned = [gambar_sorted[i] for i in filter_indices]
    fitur_cleaned = fitur_sorted[filter_indices]
    progress.progress(60)

    # Step 4: Clustering
    status_text.text("Step 4: Performing clustering...")
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(fitur_cleaned)
    labels = kmeans.labels_
    progress.progress(80)

    # Step 5: Organizing Results
    status_text.text("Step 5: Organizing clustering results...")
    gambar_kelompok = [[] for _ in range(k)]
    for idx, label in enumerate(labels):
        gambar_kelompok[label].append(gambar_cleaned[idx])
    silhouette_avg = silhouette_score(fitur_cleaned, labels)
    progress.progress(100)

    # Completion
    status_text.text("Clustering completed!")
    return gambar_kelompok, kmeans, silhouette_avg, fitur_cleaned

def plot_elbow_method(fitur):
    """Generate Elbow Method plot"""
    distortions = []
    K = range(1, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
        kmeans.fit(fitur)
        distortions.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('Jumlah Cluster (k)')
    plt.ylabel('Distortion')
    plt.title('Metode Elbow untuk Menentukan Nilai k')
    return plt

def plot_clustering_visualization(gambar_kelompok, kmeans, fitur):
    """Create clustering visualization plots"""
    fig = plt.figure(figsize=(14, 6))
    warna = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  # Extended color palette

    # Check feature dimensions for plotting
    feature_dim = fitur.shape[1]
    
    # 3D Plot (if enough features)
    if feature_dim >= 3:
        ax1 = fig.add_subplot(121, projection='3d')
        for i, gambar_cluster in enumerate(gambar_kelompok):
            fitur_cluster = np.array([ekstrak_fitur_warna(img) for img in gambar_cluster])
            ax1.scatter(fitur_cluster[:, 0], fitur_cluster[:, 1], fitur_cluster[:, 2], 
                        c=warna[i % len(warna)], label=f'Cluster {i+1}')

        centroid = kmeans.cluster_centers_
        ax1.scatter(centroid[:, 0], centroid[:, 1], centroid[:, 2], 
                    marker='*', c='purple', s=200, label='Centroid')

        ax1.set_xlabel('Rata-rata Hue')
        ax1.set_ylabel('Rata-rata Saturation')
        ax1.set_zlabel('Rata-rata Value')
        ax1.set_title('Clustering Berdasarkan Warna (3D)')
        ax1.legend()
    else:
        ax1 = fig.add_subplot(121)
        ax1.text(0.5, 0.5, 'Insufficient features for 3D plot', 
                 horizontalalignment='center', verticalalignment='center')
        ax1.set_title('3D Plot Not Available')

    # 2D Plot
    ax2 = fig.add_subplot(122)
    for i, gambar_cluster in enumerate(gambar_kelompok):
        fitur_cluster = np.array([ekstrak_fitur_warna(img) for img in gambar_cluster])
        ax2.scatter(fitur_cluster[:, 0], fitur_cluster[:, 1], 
                    c=warna[i % len(warna)], label=f'Cluster {i+1}')

    centroid = kmeans.cluster_centers_
    ax2.scatter(centroid[:, 0], centroid[:, 1], 
                marker='*', c='purple', s=200, label='Centroid')

    ax2.set_xlabel('Rata-rata Hue')
    ax2.set_ylabel('Rata-rata Saturation')
    ax2.set_title('Clustering Berdasarkan Warna (2D)')
    ax2.legend()

    plt.tight_layout()
    return plt

def main():
    st.title('Fruit Ripeness Clustering App')

    # Sidebar for parameters
    st.sidebar.header('Clustering Parameters')
    num_clusters = st.sidebar.slider('Number of Clusters', min_value=2, max_value=10, value=2)
    delta_value = st.sidebar.number_input('Delta Value', min_value=0.1, max_value=10.0, value=6.5, step=0.1)
    sample_count = st.sidebar.number_input('Number of Sample Images per Cluster', min_value=1, max_value=10, value=5)

    # File uploader
    uploaded_files = st.file_uploader("Choose fruit images", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])

    if uploaded_files:
        # Read images
        gambar = []
        for uploaded_file in uploaded_files:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            gambar.append(img)

        if st.button('Start Clustering'):
            # Perform clustering
            gambar_kelompok, kmeans, silhouette_avg, fitur_cleaned = perform_clustering(gambar, k=num_clusters, delta=delta_value)

            # Visualizations and Results
            st.subheader('Clustering Results')

            # Silhouette Score
            st.metric('Silhouette Score', f'{silhouette_avg:.4f}')

            # Elbow Method Plot
            st.subheader('Elbow Method')
            elbow_plot = plot_elbow_method(fitur_cleaned)
            st.pyplot(elbow_plot)

            # Clustering Visualization
            st.subheader('Clustering Visualization')
            cluster_plot = plot_clustering_visualization(gambar_kelompok, kmeans, fitur_cleaned)
            st.pyplot(cluster_plot)

            # Sample Images per Cluster
            st.subheader('Sample Images per Cluster')
            cols = st.columns(num_clusters)
            for i, cluster_images in enumerate(gambar_kelompok):
                with cols[i]:
                    st.write(f'Cluster {i+1}')
                    for j in range(min(sample_count, len(cluster_images))):
                        st.image(cv2.cvtColor(cluster_images[j], cv2.COLOR_BGR2RGB), use_container_width=True)

            # Cluster Summary Table
            data = {
                'Cluster': [f'Cluster {i+1}' for i in range(num_clusters)],
                'Number of Images': [len(cluster) for cluster in gambar_kelompok]
            }
            cluster_table = pd.DataFrame(data)
            st.subheader('Cluster Summary')
            st.table(cluster_table)

if __name__ == '__main__':
    main()

# streamlit run app.py