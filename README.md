# Capillary Fracturing – Image Analysis and Example Data

This repository contains the analysis scripts and example image data associated with the **capillary fracturing** experiments performed in a Hele–Shaw cell with wet granular media. The scripts process raw images to extract fracture patterns, corrected backgrounds, and skeletonised networks.

Only **lightweight example data** is included.  
The full experimental dataset is large and stored locally (not part of this repository).

---

## Repository Structure


---

## Contents

### **Analysis/scripts/**
Python scripts for:

- Background correction  
- Image preprocessing  
- Frangi/curvature filtering  
- Thresholding and segmentation  
- Skeletonisation and graph extraction  
- Parameter sweeps and batch analysis (ongoing)

All scripts are written for Python 3.X and use standard scientific libraries (`numpy`, `opencv-python`, `scikit-image`, `matplotlib`, etc.).

---

## Example Data

The folder `data/example-data/` contains a small set of raw images from a controlled-flow-rate experiment:

- Flow rate: **0.01 ml/min**  
- 8 image frames showing the early development of the capillary fracture pattern  
- Suitable for testing the analysis scripts without the large dataset

---

## Full Dataset (Excluded)

The full dataset lives in:

