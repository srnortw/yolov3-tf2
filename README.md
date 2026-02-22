# YOLOv3-TF2 Extended Pipeline
**An End-to-End Research, Training, and Edge Deployment Ecosystem**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_svg.svg)](https://test-yolov3-tf2-mkmmq7q8mm7d28euvbak7f.streamlit.app/)

This project takes a standard YOLOv3 implementation and transforms it into a full-scale MLOps pipeline. It covers everything from SQL-based data ingestion and statistical preprocessing to custom graph-mode training, evaluation via FiftyOne, and INT8 quantized edge deployment on a Raspberry Pi 3B+.

*Note: This project was originally forked from [zzh8829/yolov3-tf2](https://github.com/zzh8829/yolov3-tf2) and heavily extended to include a complete data engineering and edge inference workflow.*

---

## `yolov3-tf2.ipynb`


The heart and brain of this entire project is the **`yolov3-tf2.ipynb`** Google Colab notebook. It acts as the central orchestrator for the entire workflow. Instead of running scripts manually, this notebook automates the end-to-end experiment lifecycle:

1. **Dataset Acquisition:** Automatically pulls datasets directly from Roboflow.
2. **Pipeline Execution:** Sequentially triggers the core modules: `c_sql_d.py` $\rightarrow$ `vision_prep.py` $\rightarrow$ `train.py` $\rightarrow$ `gtruth_pred_for_fiftyone.py` $\rightarrow$ `export_tflite.py`.
3. **Evaluation Integration:** Directly embeds **FiftyOne** within the notebook environment (without breaking dependencies) to plot per-class PR curves and visually compare predicted vs. ground-truth bounding boxes.

By centralizing the execution here, the project becomes a fully reproducible, cloud-ready MLOps environment.

---

## 🔬 Deep Dive: The Underlying Modules

### 1. Data Ingestion & Storage (`c_sql_d.py`)
Triggered by the notebook, this script handles the ingestion of ZIP datasets containing images and ground-truth annotations. It routes metadata into a structured PostgreSQL database (with optional MongoDB NoSQL support), organizing it into specific tables for classes, images, and object instances.

### 2. Advanced Data Preprocessing (`vision_prep.py`)
This script builds a generator to create a `tf.dataset` with ragged tensors for flexible outputs. 
* **Statistical Splitting (Experimental):** Calculates the correlation matrix of batched RGB histograms (divided by 255 and pixel-wise(Z-score normalization)) and performs K-Means clustering (centroid=1, visualized via PCA). The closest samples to the centroid are assigned to CV/Test, while the rest go to Train/Train-CV. *(Note: This experimental approach caused some Data Mismatch errors, so default random splitting is also supported).*
* **Anchor Optimization:** Calculates optimal prior widths for the specific training dataset and compares their average IoU against the original YOLOv3 paper anchors.
* **Smart Image Processing:** This takes Z-score normalizated batched histogram moments.Applies dataset-specific filters based on batch size and normalized histogram moments, including Histogram Equalization, Gamma Correction, Median Blur, and Bilateral Filtering (to avoid blurring crucial bounding box edges).
* **Serialization:** Serializes it into GZIP-compressed TFRecords (train, traincv, cv, test).

### 3. Custom Training Loop (`train.py`)
Bypasses standard Keras fitting for ultimate control, utilizing autograd in graph mode for flexible training.
* **Cosine Learning Rate Restarts:** LR decays and restarts every batch to help the optimizer find global minima instead of getting stuck in local minima.
* **Stateless Augmentation:** Uses `tf.image.stateless` with specific seeds for reproducible augmentations (left/right flip, conditional upside-down flip based on dataset context, contrast, brightness, saturation, hue). These augmentations shift dynamically every epoch.
* **Multiscale Training:** True to the original YOLOv3 paper, the input resolution automatically changes every 10 batches.

### 4. Core Model & Dataset Utilities
* **`yolov3_tf2/dataset.py`:** Efficiently loads the GZIP-compressed TFRecord files.
* **`yolov3_tf2/models.py`:** Loads the optimized prior widths via `.npy` files outputted by `vision_prep.py`. It features a **vectorized NMS function** (detecting across the whole batch) and attaches an **anchor ID to each detection**. *Insight: Tracking anchor IDs helps analyze network behavior, e.g., higher IDs typically correspond to tiny or far-away objects.*

### 5. Ground Truth Tracking (`gtruth_pred_for_fiftyone.py`)
Outputs images, detection `.txt` files, and ground-truth `.txt` files for the train-cv, cv, and test splits. This prepares the data perfectly for the FiftyOne visualization triggered in the main notebook. *(Marked for future reliability updates).*

### 6. Edge Deployment & Quantization
Getting a heavy model onto an edge device required strict optimization:
* **`export_tflite.py`:** Converts the model using full INT8 quantization via a representative dataset to ensure the model learns the quantization bounds accurately. The graph is cut off at the `yolo_loss` function; heavy box conversion (xmin, ymin, xmax, ymax) and NMS are kept outside the graph, as including them degrades quantization quality.
* **`quantized_model_prediction_cam.py`:** Uses `ai-edge-litert` to run real-time camera object detection. 
* **`models/models_raspi3bp.py`:** The Raspberry Pi brain. Uses pure NumPy for YOLO box calculations and NMS, and OpenCV for drawing bounding boxes. 
* **Hardware Constraints:** Currently running on the newest OS for the Raspberry Pi 3B+. Due to the heavy nature of YOLOv3 and CPU constraints, it runs at a highly limited **~0.13 FPS (1 frame every 7.5 seconds)**. 

### 7. Interactive UI (`streamlit_dashb.py`)
A user-friendly dashboard built with Streamlit that imports functions from `models_raspi3bp.py`. Users can upload images for inference and dynamically adjust the **Score Threshold** and **NMS Sigma** via sliders.

---

## 📊 Datasets & Benchmarks
The pipeline was tested using an 80-10-5-5 split (train, train-cv, cv, test) utilizing the default random splitting method:

| Dataset | mAP | Link |
| :--- | :--- | :--- |
| **African Wildlife** | 0.60 | [Kaggle](https://www.kaggle.com/datasets/biancaferreira/african-wildlife) |
| **Construction Safety** | 0.35 | [Roboflow Universe](https://universe.roboflow.com/roboflow-100/construction-safety-gsnvb) |
