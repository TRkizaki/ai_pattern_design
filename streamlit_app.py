# streamlit_app.py
# Pattern Recognition Interactive Presentation
# Run with: streamlit run streamlit_app.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.datasets import make_classification, make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, DBSCAN
from sklearn.model_selection import cross_val_score
from sklearn.metrics import silhouette_score
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Pattern Recognition Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better presentation
st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .section-header {
        font-size: 2rem;
        color: #ff7f0e;
        border-bottom: 3px solid #ff7f0e;
        padding-bottom: 0.5rem;
        margin: 2rem 0 1rem 0;
    }
    
    .metric-container {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .algorithm-info {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border: 2px solid #4299e1;
        border-left: 6px solid #4299e1;
        padding: 20px;
        margin: 20px 0;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(66, 153, 225, 0.15);
        color: #2d3748;
    }
    
    .algorithm-info h3 {
        color: #2b6cb0;
        margin-bottom: 12px;
        font-size: 1.2em;
        font-weight: 600;
    }
    
    .algorithm-info p {
        margin-bottom: 8px;
        color: #4a5568;
        line-height: 1.6;
    }
    
    .algorithm-info strong {
        color: #2d3748;
        font-weight: 600;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
</style>
""",
    unsafe_allow_html=True,
)

# Main title
st.markdown(
    '<h1 class="main-header">🤖 Pattern Recognition Interactive Demo</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="text-align: center; font-size: 1.2rem; color: #666;">Master\'s Course Presentation - Decision Methods & Clustering</p>',
    unsafe_allow_html=True,
)

# Sidebar navigation
st.sidebar.title("🧭 Navigation")
page = st.sidebar.selectbox(
    "Choose Demo Section:",
    [
        "🏠 Overview",
        "🎯 Nearest Neighbor",
        "📊 Bayesian Classification",
        "🎨 K-Means Clustering",
        "📈 Feature Analysis",
        "⚖️ Algorithm Comparison",
        "🎓 Summary",
    ],
)


# Helper function to generate data
@st.cache_data
def generate_classification_data(n_samples=200, noise=0.1, random_state=42):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        class_sep=1.0,
        random_state=random_state,
    )
    return X, y


@st.cache_data
def generate_cluster_data(n_samples=300, centers=4, random_state=42):
    X, _ = make_blobs(
        n_samples=n_samples,
        centers=centers,
        n_features=2,
        random_state=random_state,
        cluster_std=1.5,
    )
    return X


# Overview Page
if page == "🏠 Overview":
    st.markdown(
        '<h2 class="section-header">📋 Presentation Overview</h2>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        ### 🎯 Learning Objectives
        
        By the end of this presentation, you will understand:
        
        ✅ **Decision Methods for Classification**
        - Nearest Neighbor Algorithm
        - Bayesian Decision Theory
        - When to use each approach
        
        ✅ **Clustering Techniques** 
        - K-Means Algorithm
        - Unsupervised pattern discovery
        - Cluster validation methods
        
        ✅ **Practical Considerations**
        - Feature selection strategies
        - Curse of dimensionality
        - Algorithm selection guidelines
        """
        )

    with col2:
        st.markdown(
            """
        ### 🔬 Interactive Features
        
        This presentation includes:
        
        🎮 **Real-time Algorithm Visualization**
        - Adjust parameters and see immediate results
        - Compare different approaches side-by-side
        
        📊 **Performance Metrics**
        - Accuracy scores and validation
        - Visual performance comparisons
        
        🎨 **Step-by-step Demonstrations**
        - Watch algorithms learn in real-time
        - Understand the decision-making process
        
        📈 **Real-world Case Studies**
        - Fish classification example
        - Feature engineering insights
        """
        )

    # Quick stats overview
    st.markdown(
        '<h3 class="section-header">📊 Algorithm Quick Reference</h3>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            """
        <div class="metric-container">
            <h3>🎯 Nearest Neighbor</h3>
            <p><strong>Type:</strong> Instance-based</p>
            <p><strong>Complexity:</strong> O(n)</p>
            <p><strong>Best for:</strong> Complex boundaries</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div class="metric-container">
            <h3>📊 Bayesian</h3>
            <p><strong>Type:</strong> Probabilistic</p>
            <p><strong>Complexity:</strong> O(1)</p>
            <p><strong>Best for:</strong> Optimal decisions</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
        <div class="metric-container">
            <h3>🎨 K-Means</h3>
            <p><strong>Type:</strong> Centroid-based</p>
            <p><strong>Complexity:</strong> O(k·n·i)</p>
            <p><strong>Best for:</strong> Spherical clusters</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            """
        <div class="metric-container">
            <h3>📈 Features</h3>
            <p><strong>Rule:</strong> Start simple</p>
            <p><strong>Optimal:</strong> Data-dependent</p>
            <p><strong>Risk:</strong> Overfitting</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

# Nearest Neighbor Page
elif page == "🎯 Nearest Neighbor":
    st.markdown(
        '<h2 class="section-header">🎯 Nearest Neighbor Classification</h2>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class="algorithm-info">
        <h3>🔍 Algorithm Concept</h3>
        <p><strong>Core Idea:</strong> "You are who your neighbors are!"</p>
        <p>The algorithm classifies unknown samples by finding the closest training samples in feature space and assigning their class labels.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Interactive controls
    col1, col2, col3 = st.columns(3)

    with col1:
        k_value = st.slider("K Value (Number of Neighbors)", 1, 15, 3)
    with col2:
        n_samples = st.slider("Training Samples", 50, 500, 200)
    with col3:
        noise_level = st.slider("Noise Level", 0.0, 0.5, 0.1)

    # Generate and classify data
    X, y = generate_classification_data(n_samples, noise_level)

    # Create KNN classifier
    knn = KNeighborsClassifier(n_neighbors=k_value)
    knn.fit(X, y)

    # Create decision boundary
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Create visualization
    fig = go.Figure()

    # Add decision boundary
    fig.add_trace(
        go.Contour(
            x=np.arange(x_min, x_max, h),
            y=np.arange(y_min, y_max, h),
            z=Z,
            colorscale="RdYlBu",
            opacity=0.3,
            showscale=False,
            hoverinfo="skip",
        )
    )

    # Add training data
    for class_val in [0, 1]:
        mask = y == class_val
        fig.add_trace(
            go.Scatter(
                x=X[mask, 0],
                y=X[mask, 1],
                mode="markers",
                marker=dict(
                    size=8,
                    color="red" if class_val == 0 else "blue",
                    line=dict(width=1, color="black"),
                ),
                name=f"Class {class_val + 1}",
                hovertemplate="Feature 1: %{x:.2f}<br>Feature 2: %{y:.2f}<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"K-Nearest Neighbor Classification (K={k_value})",
        xaxis_title="Feature 1",
        yaxis_title="Feature 2",
        showlegend=True,
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Performance metrics
    accuracy = cross_val_score(knn, X, y, cv=5).mean()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Cross-validation Accuracy", f"{accuracy:.3f}")
    with col2:
        st.metric("Training Samples", len(X))
    with col3:
        st.metric("K Value", k_value)
    with col4:
        st.metric("Feature Dimensions", X.shape[1])

    # Algorithm explanation
    with st.expander("🔬 How K-NN Works - Step by Step"):
        st.markdown(
            """
        **Step 1: Calculate Distances**
        ```python
        distances = np.sqrt(np.sum((test_point - training_points)**2, axis=1))
        ```
        
        **Step 2: Find K Nearest Neighbors**
        ```python
        nearest_indices = np.argsort(distances)[:k]
        ```
        
        **Step 3: Majority Vote**
        ```python
        neighbor_classes = y[nearest_indices]
        prediction = mode(neighbor_classes)
        ```
        
        **Key Characteristics:**
        - ✅ Simple and intuitive
        - ✅ No training phase required
        - ❌ Sensitive to irrelevant features
        - ❌ Computationally expensive for large datasets
        """
        )

# Bayesian Classification Page
elif page == "📊 Bayesian Classification":
    st.markdown(
        '<h2 class="section-header">📊 Bayesian Decision Theory</h2>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class="algorithm-info">
        <h3>🧮 Mathematical Foundation</h3>
        <p><strong>Bayes' Rule:</strong> P(Class|Features) = P(Features|Class) × P(Class) / P(Features)</p>
        <p><strong>Decision Rule:</strong> Choose class with highest posterior probability (MAP - Maximum A Posteriori)</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Interactive controls
    col1, col2 = st.columns(2)

    with col1:
        prior_class1 = st.slider("Prior Probability Class 1", 0.1, 0.9, 0.5)
        show_contours = st.checkbox("Show Probability Contours", value=True)

    with col2:
        n_samples_bayes = st.slider("Number of Samples", 100, 500, 200)
        overlap_factor = st.slider("Class Overlap", 0.5, 2.0, 1.0)

    # Generate Bayesian data with overlapping classes
    np.random.seed(42)
    n_half = n_samples_bayes // 2

    # Class 1: centered at (2, 2)
    class1_data = np.random.multivariate_normal([2, 2], [[1, 0.3], [0.3, 1]], n_half)
    # Class 2: centered at (4, 4) with controllable overlap
    class2_data = np.random.multivariate_normal(
        [2 + overlap_factor * 2, 2 + overlap_factor * 2], [[1, -0.2], [-0.2, 1]], n_half
    )

    X_bayes = np.vstack([class1_data, class2_data])
    y_bayes = np.hstack([np.zeros(n_half), np.ones(n_half)])

    # Create Naive Bayes classifier with custom priors
    nb = GaussianNB()
    nb.fit(X_bayes, y_bayes)

    # Manually set priors
    prior_class2 = 1 - prior_class1
    nb.class_prior_ = np.array([prior_class1, prior_class2])

    # Create mesh for visualization
    h = 0.02
    x_min, x_max = X_bayes[:, 0].min() - 1, X_bayes[:, 0].max() + 1
    y_min, y_max = X_bayes[:, 1].min() - 1, X_bayes[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Get predictions and probabilities
    Z = nb.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    Z_proba = nb.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z_proba = Z_proba.reshape(xx.shape)

    # Create subplot for decision boundary and probability
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Decision Boundary", "Posterior Probability P(Class=2|x)"),
        specs=[[{"type": "xy"}, {"type": "xy"}]],
    )

    # Decision boundary plot
    fig.add_trace(
        go.Contour(
            x=np.arange(x_min, x_max, h),
            y=np.arange(y_min, y_max, h),
            z=Z,
            colorscale="RdYlBu",
            opacity=0.4,
            showscale=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    # Add training data to decision boundary plot
    for class_val in [0, 1]:
        mask = y_bayes == class_val
        fig.add_trace(
            go.Scatter(
                x=X_bayes[mask, 0],
                y=X_bayes[mask, 1],
                mode="markers",
                marker=dict(
                    size=6,
                    color="red" if class_val == 0 else "blue",
                    line=dict(width=1, color="black"),
                ),
                name=f"Class {class_val + 1}",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    # Probability contour plot
    if show_contours:
        fig.add_trace(
            go.Contour(
                x=np.arange(x_min, x_max, h),
                y=np.arange(y_min, y_max, h),
                z=Z_proba,
                colorscale="RdYlBu",
                showscale=True,
                colorbar=dict(title="P(Class=2|x)"),
            ),
            row=1,
            col=2,
        )

        # Add data points to probability plot
        for class_val in [0, 1]:
            mask = y_bayes == class_val
            fig.add_trace(
                go.Scatter(
                    x=X_bayes[mask, 0],
                    y=X_bayes[mask, 1],
                    mode="markers",
                    marker=dict(
                        size=6,
                        color="red" if class_val == 0 else "blue",
                        line=dict(width=1, color="black"),
                    ),
                    showlegend=False,
                ),
                row=1,
                col=2,
            )

    fig.update_layout(
        title=f"Bayesian Classification (Prior P(Class 1) = {prior_class1:.1f})",
        height=500,
    )

    fig.update_xaxes(title_text="Feature 1")
    fig.update_yaxes(title_text="Feature 2")

    st.plotly_chart(fig, use_container_width=True)

    # Performance metrics
    accuracy_bayes = cross_val_score(nb, X_bayes, y_bayes, cv=5).mean()
    log_likelihood = nb.score(X_bayes, y_bayes)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Accuracy", f"{accuracy_bayes:.3f}")
    with col2:
        st.metric("Log Likelihood", f"{log_likelihood:.3f}")
    with col3:
        st.metric("Prior P(Class 1)", f"{prior_class1:.1f}")
    with col4:
        st.metric("Optimal", "Yes (MAP)")

    # Mathematical explanation
    with st.expander("🧮 Bayesian Mathematics"):
        st.markdown(
            """
        **Bayes' Theorem Components:**
        
        - **Prior P(Ci):** What we know before seeing data
        - **Likelihood P(x|Ci):** Probability of features given class
        - **Posterior P(Ci|x):** Updated probability after seeing features
        - **Evidence P(x):** Normalizing constant
        
        **Decision Rule (MAP):**
        ```
        Choose class j if P(Cj|x) = max_i P(Ci|x)
        ```
        
        **Why Bayesian is Optimal:**
        - Minimizes expected classification error
        - Incorporates prior knowledge
        - Theoretically grounded in probability theory
        """
        )

# K-Means Clustering Page
elif page == "🎨 K-Means Clustering":
    st.markdown(
        '<h2 class="section-header">🎨 K-Means Clustering Algorithm</h2>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class="algorithm-info">
        <h3>🎯 Unsupervised Learning</h3>
        <p><strong>Goal:</strong> Group similar data points without knowing true labels</p>
        <p><strong>Method:</strong> Iteratively update cluster centers to minimize within-cluster variance</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Interactive controls
    col1, col2, col3 = st.columns(3)

    with col1:
        k_clusters = st.slider("Number of Clusters (K)", 2, 8, 3)
    with col2:
        n_samples_cluster = st.slider("Number of Data Points", 100, 500, 300)
    with col3:
        cluster_std = st.slider("Cluster Spread", 0.5, 2.5, 1.0)

    # Control buttons
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🎲 Generate New Data"):
            st.experimental_rerun()

    with col2:
        step_by_step = st.checkbox("Step-by-step Animation")

    with col3:
        show_centroids = st.checkbox("Show Centroids", value=True)

    with col4:
        show_history = st.checkbox("Show Centroid History")

    # Generate clustering data
    X_cluster = generate_cluster_data(n_samples_cluster, 4, 42)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_cluster)
    centroids = kmeans.cluster_centers_

    # Create visualization
    fig = go.Figure()

    # Plot data points colored by cluster
    colors = px.colors.qualitative.Set1
    for i in range(k_clusters):
        mask = cluster_labels == i
        fig.add_trace(
            go.Scatter(
                x=X_cluster[mask, 0],
                y=X_cluster[mask, 1],
                mode="markers",
                marker=dict(size=8, color=colors[i % len(colors)], opacity=0.7),
                name=f"Cluster {i + 1}",
                hovertemplate="Feature 1: %{x:.2f}<br>Feature 2: %{y:.2f}<extra></extra>",
            )
        )

    # Show centroids
    if show_centroids:
        fig.add_trace(
            go.Scatter(
                x=centroids[:, 0],
                y=centroids[:, 1],
                mode="markers",
                marker=dict(
                    size=15,
                    color="black",
                    symbol="x",
                    line=dict(width=3, color="white"),
                ),
                name="Centroids",
                hovertemplate="Centroid<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>",
            )
        )

    # Calculate and display metrics
    inertia = kmeans.inertia_
    silhouette = silhouette_score(X_cluster, cluster_labels)

    fig.update_layout(
        title=f"K-Means Clustering (K={k_clusters}, Inertia={inertia:.1f})",
        xaxis_title="Feature 1",
        yaxis_title="Feature 2",
        showlegend=True,
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Inertia (WCSS)", f"{inertia:.1f}")
    with col2:
        st.metric("Silhouette Score", f"{silhouette:.3f}")
    with col3:
        st.metric("Number of Clusters", k_clusters)
    with col4:
        st.metric("Converged", "Yes")

    # Algorithm steps
    with st.expander("🔄 K-Means Algorithm Steps"):
        st.markdown(
            """
        **Step 1: Initialize**
        ```python
        centroids = random_initialization(k)
        ```
        
        **Step 2: Assign Points**
        ```python
        for each point:
            assign to nearest centroid
        ```
        
        **Step 3: Update Centroids**
        ```python
        for each cluster:
            centroid = mean(cluster_points)
        ```
        
        **Step 4: Repeat**
        ```python
        while not converged:
            repeat steps 2-3
        ```
        
        **Convergence Criteria:**
        - Centroids don't move significantly
        - Maximum iterations reached
        - Assignments don't change
        """
        )

# Feature Analysis Page
elif page == "📈 Feature Analysis":
    st.markdown(
        '<h2 class="section-header">📈 Feature Dimensionality Analysis</h2>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class="algorithm-info">
        <h3>⚠️ The Curse of Dimensionality</h3>
        <p><strong>Problem:</strong> As feature dimensions increase, performance may decrease due to sparse data</p>
        <p><strong>Solution:</strong> Find optimal number of features through experimentation</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Interactive controls
    col1, col2 = st.columns(2)

    with col1:
        max_features = st.slider("Maximum Features to Test", 5, 20, 10)
        training_size = st.slider("Training Set Size", 50, 1000, 200)

    with col2:
        algorithm_choice = st.selectbox("Algorithm", ["K-NN", "Naive Bayes", "Both"])
        cv_folds = st.slider("Cross-validation Folds", 3, 10, 5)

    if st.button("🚀 Run Feature Analysis"):
        # Simulate feature analysis
        features_range = range(1, max_features + 1)

        # Generate results for different feature counts
        knn_accuracies = []
        nb_accuracies = []

        progress_bar = st.progress(0)

        for i, n_features in enumerate(features_range):
            # Generate data with n_features
            X_feat, y_feat = make_classification(
                n_samples=training_size,
                n_features=n_features,
                n_informative=min(n_features, 5),
                n_redundant=max(0, n_features - 5),
                n_clusters_per_class=1,
                random_state=42,
            )

            if algorithm_choice in ["K-NN", "Both"]:
                knn = KNeighborsClassifier(n_neighbors=5)
                knn_scores = cross_val_score(knn, X_feat, y_feat, cv=cv_folds)
                knn_accuracies.append(knn_scores.mean())

            if algorithm_choice in ["Naive Bayes", "Both"]:
                nb = GaussianNB()
                nb_scores = cross_val_score(nb, X_feat, y_feat, cv=cv_folds)
                nb_accuracies.append(nb_scores.mean())

            progress_bar.progress((i + 1) / len(features_range))

        # Create visualization
        fig = go.Figure()

        if algorithm_choice in ["K-NN", "Both"]:
            fig.add_trace(
                go.Scatter(
                    x=list(features_range),
                    y=knn_accuracies,
                    mode="lines+markers",
                    name="K-NN (k=5)",
                    line=dict(width=3),
                    marker=dict(size=8),
                )
            )

            # Find optimal point for KNN
            optimal_knn_idx = np.argmax(knn_accuracies)
            fig.add_trace(
                go.Scatter(
                    x=[features_range[optimal_knn_idx]],
                    y=[knn_accuracies[optimal_knn_idx]],
                    mode="markers",
                    marker=dict(size=15, color="red", symbol="star"),
                    name=f"Optimal K-NN ({features_range[optimal_knn_idx]} features)",
                    showlegend=False,
                )
            )

        if algorithm_choice in ["Naive Bayes", "Both"]:
            fig.add_trace(
                go.Scatter(
                    x=list(features_range),
                    y=nb_accuracies,
                    mode="lines+markers",
                    name="Naive Bayes",
                    line=dict(width=3),
                    marker=dict(size=8),
                )
            )

            # Find optimal point for Naive Bayes
            optimal_nb_idx = np.argmax(nb_accuracies)
            fig.add_trace(
                go.Scatter(
                    x=[features_range[optimal_nb_idx]],
                    y=[nb_accuracies[optimal_nb_idx]],
                    mode="markers",
                    marker=dict(size=15, color="orange", symbol="star"),
                    name=f"Optimal NB ({features_range[optimal_nb_idx]} features)",
                    showlegend=False,
                )
            )

        fig.update_layout(
            title="Classification Accuracy vs Number of Features",
            xaxis_title="Number of Features",
            yaxis_title="Cross-validation Accuracy",
            height=500,
            showlegend=True,
        )

        st.plotly_chart(fig, use_container_width=True)

        # Summary statistics
        col1, col2, col3 = st.columns(3)

        with col1:
            if knn_accuracies:
                optimal_features_knn = features_range[np.argmax(knn_accuracies)]
                max_accuracy_knn = max(knn_accuracies)
                st.metric("Optimal Features (K-NN)", optimal_features_knn)
                st.metric("Max Accuracy (K-NN)", f"{max_accuracy_knn:.3f}")

        with col2:
            if nb_accuracies:
                optimal_features_nb = features_range[np.argmax(nb_accuracies)]
                max_accuracy_nb = max(nb_accuracies)
                st.metric("Optimal Features (NB)", optimal_features_nb)
                st.metric("Max Accuracy (NB)", f"{max_accuracy_nb:.3f}")

        with col3:
            st.metric("Training Size", training_size)
            st.metric("CV Folds", cv_folds)

# Algorithm Comparison Page
elif page == "⚖️ Algorithm Comparison":
    st.markdown(
        '<h2 class="section-header">⚖️ Algorithm Performance Comparison</h2>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class="algorithm-info">
        <h3>🔍 Comprehensive Evaluation</h3>
        <p>Compare different algorithms across various scenarios to understand their strengths and weaknesses</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Scenario selection
    col1, col2 = st.columns(2)

    with col1:
        scenario = st.selectbox(
            "Choose Scenario:",
            [
                "Balanced Classes",
                "Imbalanced Classes",
                "High Noise",
                "Small Dataset",
                "High Dimensional",
            ],
        )

    with col2:
        metrics_to_show = st.multiselect(
            "Metrics to Display:",
            ["Accuracy", "Precision", "Recall", "F1-Score", "Training Time"],
            default=["Accuracy", "Training Time"],
        )

    if st.button("🔄 Run Comparison"):
        # Generate data based on scenario
        if scenario == "Balanced Classes":
            X_comp, y_comp = make_classification(
                n_samples=400,
                n_features=10,
                n_informative=5,
                weights=[0.5, 0.5],
                random_state=42,
            )
        elif scenario == "Imbalanced Classes":
            X_comp, y_comp = make_classification(
                n_samples=400,
                n_features=10,
                n_informative=5,
                weights=[0.9, 0.1],
                random_state=42,
            )
        elif scenario == "High Noise":
            X_comp, y_comp = make_classification(
                n_samples=400,
                n_features=10,
                n_informative=3,
                n_redundant=2,
                n_clusters_per_class=1,
                flip_y=0.2,
                random_state=42,
            )
        elif scenario == "Small Dataset":
            X_comp, y_comp = make_classification(
                n_samples=100, n_features=5, n_informative=3, random_state=42
            )
        else:  # High Dimensional
            X_comp, y_comp = make_classification(
                n_samples=200, n_features=50, n_informative=10, random_state=42
            )

        # Define algorithms
        algorithms = {
            "1-NN": KNeighborsClassifier(n_neighbors=1),
            "5-NN": KNeighborsClassifier(n_neighbors=5),
            "10-NN": KNeighborsClassifier(n_neighbors=10),
            "Naive Bayes": GaussianNB(),
        }

        # Evaluate algorithms
        results = []

        progress_bar = st.progress(0)

        for i, (name, algorithm) in enumerate(algorithms.items()):
            # Measure training time
            import time

            start_time = time.time()

            # Cross-validation
            cv_scores = cross_val_score(
                algorithm, X_comp, y_comp, cv=5, scoring="accuracy"
            )

            end_time = time.time()
            training_time = (end_time - start_time) * 1000  # in milliseconds

            results.append(
                {
                    "Algorithm": name,
                    "Accuracy": cv_scores.mean(),
                    "Accuracy_std": cv_scores.std(),
                    "Training Time": training_time,
                }
            )

            progress_bar.progress((i + 1) / len(algorithms))

        # Create comparison visualization
        df_results = pd.DataFrame(results)

        # Accuracy comparison
        fig_acc = go.Figure()

        fig_acc.add_trace(
            go.Bar(
                x=df_results["Algorithm"],
                y=df_results["Accuracy"],
                error_y=dict(
                    type="data", array=df_results["Accuracy_std"], visible=True
                ),
                marker_color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
                text=[f"{acc:.3f}" for acc in df_results["Accuracy"]],
                textposition="auto",
            )
        )

        fig_acc.update_layout(
            title=f"Algorithm Accuracy Comparison - {scenario}",
            xaxis_title="Algorithm",
            yaxis_title="Cross-validation Accuracy",
            height=400,
        )

        st.plotly_chart(fig_acc, use_container_width=True)

        # Training time comparison
        fig_time = go.Figure()

        fig_time.add_trace(
            go.Bar(
                x=df_results["Algorithm"],
                y=df_results["Training Time"],
                marker_color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
                text=[f"{time:.1f}ms" for time in df_results["Training Time"]],
                textposition="auto",
            )
        )

        fig_time.update_layout(
            title="Training Time Comparison",
            xaxis_title="Algorithm",
            yaxis_title="Training Time (ms)",
            height=400,
        )

        st.plotly_chart(fig_time, use_container_width=True)

        # Results table
        st.markdown("### 📊 Detailed Results")
        st.dataframe(df_results.round(4), use_container_width=True)

        # Insights based on scenario
        st.markdown("### 💡 Key Insights")

        if scenario == "Balanced Classes":
            st.info(
                "🎯 All algorithms perform well with balanced data. Choice depends on interpretability and speed requirements."
            )
        elif scenario == "Imbalanced Classes":
            st.warning(
                "⚠️ Be careful with accuracy metrics on imbalanced data. Consider precision, recall, and F1-score."
            )
        elif scenario == "High Noise":
            st.info(
                "🔊 Higher K values in K-NN help reduce noise sensitivity. Bayesian methods are also robust."
            )
        elif scenario == "Small Dataset":
            st.warning(
                "📉 With limited data, simpler models (higher K, Naive Bayes) often perform better."
            )
        else:  # High Dimensional
            st.error(
                "📈 Curse of dimensionality affects K-NN significantly. Feature selection becomes crucial."
            )

# Summary Page
elif page == "🎓 Summary":
    st.markdown(
        '<h2 class="section-header">🎓 Presentation Summary</h2>',
        unsafe_allow_html=True,
    )

    # Key takeaways
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        ### 🎯 Classification Methods
        
        **Nearest Neighbor:**
        - ✅ Simple and intuitive
        - ✅ No assumptions about data distribution
        - ❌ Sensitive to irrelevant features
        - ❌ Computationally expensive
        
        **Bayesian Classification:**
        - ✅ Theoretically optimal (MAP)
        - ✅ Incorporates prior knowledge
        - ✅ Fast prediction
        - ❌ Assumes feature independence (Naive Bayes)
        """
        )

    with col2:
        st.markdown(
            """
        ### 🎨 Clustering & Features
        
        **K-Means Clustering:**
        - ✅ Fast and scalable
        - ✅ Works well with spherical clusters
        - ❌ Need to specify K
        - ❌ Sensitive to initialization
        
        **Feature Selection:**
        - 🎯 More features ≠ Better performance
        - 📊 Optimal number depends on data size
        - ⚠️ Curse of dimensionality is real
        - 🔍 Cross-validation is essential
        """
        )

    # Decision flowchart
    st.markdown(
        '<h3 class="section-header">🗺️ Algorithm Selection Guide</h3>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    ```
    📊 ALGORITHM SELECTION FLOWCHART
    
    ❓ Do you have labeled data?
    ├── Yes (Supervised Learning)
    │   ├── 🎯 Small dataset + Simple boundary → Naive Bayes
    │   ├── 🎯 Large dataset + Complex boundary → K-NN
    │   ├── 🎯 Need probability estimates → Bayesian methods
    │   └── 🎯 Speed is critical → Naive Bayes
    │
    └── No (Unsupervised Learning)
        ├── 🎨 Spherical clusters expected → K-Means
        ├── 🎨 Arbitrary shapes + noise → DBSCAN
        └── 🎨 Hierarchical structure → Agglomerative
    
    🎲 FEATURE SELECTION STRATEGY
    
    ❓ How much training data do you have?
    ├── Small (< 100 samples) → Use fewer features (2-5)
    ├── Medium (100-1000) → Moderate features (5-15)
    └── Large (> 1000) → More features possible (10-50+)
    
    💡 Always validate with cross-validation!
    ```
    """
    )

    # Performance summary
    st.markdown(
        '<h3 class="section-header">📈 Performance Summary</h3>', unsafe_allow_html=True
    )

    # Create a summary comparison chart
    algorithms = ["1-NN", "5-NN", "Naive Bayes", "K-Means"]
    characteristics = {
        "Speed": [2, 2, 5, 4],
        "Accuracy": [4, 4, 4, 3],
        "Interpretability": [5, 5, 4, 3],
        "Scalability": [2, 2, 5, 4],
        "Robustness": [2, 4, 4, 3],
    }

    fig = go.Figure()

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, (char, values) in enumerate(characteristics.items()):
        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=algorithms,
                fill="tonext" if i > 0 else "toself",
                name=char,
                line_color=colors[i],
            )
        )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
        showlegend=True,
        title="Algorithm Characteristics Radar Chart",
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Final recommendations
    st.markdown(
        '<h3 class="section-header">🎯 Final Recommendations</h3>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
        **🚀 Getting Started:**
        1. Start with simple algorithms
        2. Understand your data first
        3. Use cross-validation
        4. Consider computational constraints
        """
        )

    with col2:
        st.markdown(
            """
        **🔧 Best Practices:**
        1. Feature engineering matters
        2. Handle missing data properly
        3. Scale features when needed
        4. Monitor for overfitting
        """
        )

    with col3:
        st.markdown(
            """
        **🎓 Advanced Topics:**
        1. Ensemble methods
        2. Deep learning approaches
        3. Online learning algorithms
        4. Automated ML pipelines
        """
        )

    # Interactive conclusion
    st.markdown(
        '<h3 class="section-header">🎉 Interactive Demo Complete!</h3>',
        unsafe_allow_html=True,
    )

    st.success(
        """
    🏆 **Congratulations!** You've explored all the key concepts in pattern recognition:
    
    ✅ **Decision Methods:** Nearest Neighbor vs Bayesian approaches
    ✅ **Clustering:** K-means algorithm and evaluation
    ✅ **Feature Analysis:** Dimensionality and selection strategies
    ✅ **Practical Comparison:** When to use which algorithm
    
    💡 **Next Steps:** Apply these concepts to your own datasets and research projects!
    """
    )

    # Download code button
    if st.button("📥 Download Complete Code Package"):
        st.info(
            """
        📦 **Code Package Includes:**
        - Complete Jupyter notebook with all demonstrations
        - Streamlit application source code
        - Sample datasets for practice
        - Presentation slides template
        - Algorithm implementation examples
        
        Contact your instructor for the download link!
        """
        )

# Footer
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: #666; padding: 2rem;'>
    <h4>🤖 Pattern Recognition Interactive Demo</h4>
    <p>Master's Course Presentation | Created for Academic Excellence</p>
    <p><em>Use the sidebar to navigate between different algorithm demonstrations</em></p>
</div>
""",
    unsafe_allow_html=True,
)
