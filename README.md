# Neural Network Image Analysis — Rock Image Embeddings & Classification

This project analyzes grayscale rock images using **dimensionality reduction**, **clustering**, and a **CNN classifier**.  
In addition to modeling, the project compares learned image embeddings against **human-provided 8D feature ratings** using **Procrustes analysis**.

---

## What this project does

### 1) Dimensionality Reduction + Visualization
- Uses **PCA** to reconstruct images and quantify variance retained (e.g., 90% variance threshold).
- Builds 2D visualizations of image structure using:
  - **PCA**
  - **t-SNE**
  - **LLE**
  - **MDS**
- Produces category-colored scatter plots (3 rock classes inferred from filename prefix: `I`, `M`, `S`).

### 2) Human Similarity Alignment (Embedding Validation)
- Loads human-provided **8-dimensional feature representations** for each image:
  - `mds_360.txt` → 360-image set (8D per image)
  - `mds_120.txt` → 120-image set (8D per image)
- Generates 8D embeddings from images using PCA / t-SNE / LLE / MDS, then:
  - compares learned vs human embeddings using **Procrustes disparity**
  - computes **correlations across embedding dimensions**

### 3) Unsupervised Clustering
- Applies clustering on image embeddings:
  - **K-Means**
  - **Expectation-Maximization (Gaussian Mixture)**
- Evaluates clustering quality (e.g., cluster alignment vs true categories).
- Generates synthetic samples from EM in the original image space via inverse transform (when PCA is used).

### 4) CNN Classification (Train on 360, Validate on 120)
- Trains a **CNN** to classify rock images into 3 categories (`I/M/S`) inferred from filenames.
- Uses:
  - training set: **360 Rocks**
  - validation set: **120 Rocks**
- Includes:
  - training time measurement
  - learning curves (loss + accuracy)
  - parameter count reporting
  - last pre-softmax layer of **8 neurons** (for representation comparison / compact embedding)

---

## Data

### Folder layout (recommended)
Place your data like this:
data/
360_rocks/                  # images for training / analysis
120_rocks/                  # validation images
human_ratings/
mds_360.txt               # 360 x 8 human feature ratings
mds_120.txt               # 120 x 8 human feature ratings
categorization_data/      # (your folder of text files)
direct_dimension_ratings/ # (your folder of text files)

**Notes**
- The rock category is inferred from the first character of the image filename: `I`, `M`, `S`.
- Human rating files provide 8 features per image in the same order as images loaded for each folder.

---

## Repo Structure (recommended)
```
├── notebooks/
│   └── Main.ipynb
├── src/
│   ├── load_data.py          # load images + labels (I/M/S)
│   ├── dim_reduce.py         # PCA, t-SNE, LLE, MDS helpers
│   ├── clustering.py         # KMeans + GMM utilities
│   ├── procrustes_eval.py    # procrustes + correlation tables
│   └── cnn_train.py          # CNN training + plots + param counts
├── data/                     # (not committed if large)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

### 1) Create environment and install dependencies
```bash
python -m venv .venv
source .venv/bin/activate      # (mac/linux)
# .venv\Scripts\activate       # (windows)
pip install -r requirements.txt
```

### 2) Run notebook
```bash
jupyter notebook
```

## Results
- PCA components needed for 90% variance: [fill]
- CNN validation accuracy (120 set): [fill]
- Best Procrustes disparity (closest to human ratings): [fill]
- Best clustering approach: [fill]

## Limitations
- Results depend on consistent image ordering when comparing with human rating files.
- t-SNE is sensitive to perplexity / learning rate; results can vary across runs.
- Clustering evaluation is limited by the simplicity of filename-derived labels.

## Next Steps
- Add MLflow logging for CNN experiments (epochs, LR, metrics).
- Add a lightweight CLI (python -m src.cnn_train --epochs 30) for reproducible training.
- Add model error analysis (confusion matrix + misclassified images gallery).
