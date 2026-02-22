### Testing on the streamlit:
https://test-yolov3-tf2-mkmmq7q8mm7d28euvbak7f.streamlit.app/

# YOLOv3-TF2 Extended Pipeline

An extended and research-oriented implementation of YOLOv3 in TensorFlow 2,
including a complete data engineering pipeline, statistical dataset analysis,
optimized training procedure, evaluation tooling, and edge deployment on Raspberry Pi.

---

## Key Features

- SQL-based dataset ingestion (PostgreSQL, optional MongoDB support)
- Statistical dataset analysis and clustering-based splitting
- Anchor prior optimization with IoU comparison
- TFRecord generation with GZIP compression
- Custom training loop with cosine learning rate restarts
- Stateless augmentation for reproducibility
- Multiscale training (YOLOv3 paper-compliant)
- Vectorized Non-Maximum Suppression (NMS)
- FiftyOne integration for evaluation and PR curves
- Full INT8 TFLite quantization
- Raspberry Pi 3B+ edge deployment
- Streamlit-based inference dashboard
- End-to-end Google Colab workflow notebook

---

## System Architecture

1. Dataset ingestion and SQL storage  
2. Statistical preprocessing and dataset splitting  
3. TFRecord serialization  
4. Model training  
5. Evaluation and visualization  
6. Edge deployment (TFLite + Raspberry Pi)

---

## Dataset Pipeline

### c_sql_d.py

- Accepts ZIP datasets containing images and annotations
- Stores metadata in structured SQL tables
- Supports PostgreSQL (MongoDB extension optional)

### vision_prep.py

- Extracts ZIP datasets
- Converts bounding boxes to (xmin, ymin, xmax, ymax)
- Computes histogram moments
- Applies Z-score normalization
- Computes correlation matrix of rgb histograms
- Performs KMeans clustering
- Splits dataset into Train / Train-CV / CV / Test( there are specific default random splitting option and k means version(this is just expiremental not that good.It causes missclassified data error i think)
- Computes optimized anchor priors
- Compares average IoU with original YOLOv3 anchors
- Applies image preprocessing:(it uses batch size and normalized histogram moments for the custom filtering (this is experimental too))
  - Histogram equalization
  - Gamma correction
  - Median blur
  - Bilateral filtering
- Serializes dataset to GZIP-compressed TFRecords

---

## Training

### train.py

- Custom TensorFlow 2 training loop (graph mode)
- Cosine learning rate decay with restarts (batch-wise)
- Multiscale training (resolution changes every 10 batches)
- Stateless augmentation:
  - Horizontal flip
  - Conditional vertical flip
  - Brightness / Contrast / Saturation / Hue adjustments

---

## Google Colab Workflow

### yolov3-tf2.ipynb

End-to-end experimental notebook that:

- Pulls dataset from Roboflow
- Executes full pipeline:
  - c_sql_d.py
  - vision_prep.py
  - train.py
  - gtruth_pred_for_fiftyone.py
  - export_tflite.py
- Integrates FiftyOne for:
  - PR curve visualization
  - Bounding box inspection
  - Ground truth vs prediction comparison

Enables reproducible experimentation and cloud-based training.

---

## Evaluation

### FiftyOne Integration

- Exports predictions and ground truths
- Visualizes bounding boxes and per-class PR curves
- Enables interactive dataset inspection



- African Animals Dataset : 0.6 mAP(traincv-cv-test,80-10-5-5,default random splitting) : https://www.kaggle.com/datasets/biancaferreira/african-wildlife
- Construction Safety Dataset : 0.35 mAP(traincv-cv-test,80-10-5-5,default random splitting) : https://universe.roboflow.com/roboflow-100/construction-safety-gsnvb
  

---

## Edge Deployment

### export_tflite.py

- Full INT8 quantization
- Representative dataset calibration
- Lightweight post-processing outside model graph

### Raspberry Pi Inference

- NumPy-based bounding box processing
- NMS implemented in NumPy
- OpenCV visualization
- Current performance: ~0.13 FPS (Raspberry Pi 3B+ CPU constraint)

---

## Streamlit Dashboard

- Upload image for inference
- Adjustable confidence threshold
- Adjustable NMS sigma
- Visualization of detections

---

## Future Improvements

- YOLOv3-Tiny edge optimization
- Structured benchmarking
- Automated anchor ablation study
- Improved evaluation export reliability
- Embedded system profiling

---

## Acknowledgements

Based on zzh8829/yolov3-tf2  
Extended and engineered by Serkan Srgvc
