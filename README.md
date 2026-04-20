# 🏙️ Cape Town Urban Change Detection Compliance Monitoring System
A Human-in-the-Loop GeoAI tool for detecting unauthorized urban development in Khayelitsha, Cape Town. This project combines Sentinel-2 satellite imagery, official Cape Town GIS layers, change detection, rule-based compliance, and machine learning to help urban planners identify and prioritize potential land-use violations.

Focus Area: Khayelitsha (Informal Settlement Expansion)

---
Dashboard Preview

<p align="center">
  <img src="https://github.com/frankraDIUM/Cape-Town-Urban-Change-Detection-Compliance-Monitoring-System/blob/main/urban.gif" />
</p>

---

Review System
<p align="center">
  <img src="https://github.com/frankraDIUM/Cape-Town-Urban-Change-Detection-Compliance-Monitoring-System/blob/main/urban1.png" />
</p>

ML Lab
<p align="center">
  <img src="https://github.com/frankraDIUM/Cape-Town-Urban-Change-Detection-Compliance-Monitoring-System/blob/main/urban2.png" />
</p>

Evaluation
<p align="center">
  <img src="https://github.com/frankraDIUM/Cape-Town-Urban-Change-Detection-Compliance-Monitoring-System/blob/main/urban3.png" />
</p>

Inspection Priority
<p align="center">
  <img src="https://github.com/frankraDIUM/Cape-Town-Urban-Change-Detection-Compliance-Monitoring-System/blob/main/urban4.png" />
</p>
---


*1. Project Objective*
Develop a human-in-the-loop GeoAI system to detect and prioritize potential unauthorized building developments and land-use violations using:

  - Sentinel-2 satellite imagery (10m resolution)
  - Official Cape Town GIS layers (buildings, zoning)
  - Change detection and  rule-based compliance engine
  - ML feedback loop with active learning elements
  - Interactive review interface for urban planners

*2. Study Area*

  - AOI: Khayelitsha (bounding box: 18.65°E to 18.70°E, 34.065°S to 34.015°S)
  - CRS: EPSG:32734 (UTM Zone 34S) for accurate area/distance calculations
  - Time periods: t1 = Jan 2022, t2 = Jan 2025

*3. Technology Stack*

  - Python, Anaconda environment
  - Geospatial: GeoPandas, Rasterio, Shapely, Folium, Streamlit-Folium
  - Imagery: odc.stac + Planetary Computer (Sentinel-2 L2A)
  - ML: scikit-learn RandomForestClassifier
  - Database: SQLite (review_log table)
  - Frontend: Streamlit dashboard with interactive Folium map

*4. Methodology & Key Components*
Phase 1 – Data Acquisition & Preprocessing

  - Clipped official building footprints and zoning layers to AOI
  - Reprojected to EPSG:32734
  - Loaded 86,977 buildings and 44,546 zoning polygons

Phase 2 – Sentinel-2 Change Detection

  - Downloaded and processed Sentinel-2 L2A scenes (2022 & 2025)
  - Applied SCL cloud/shadow masking
  - Created median composites
  - Implemented NDVI-based built-up proxy change detection
  - Morphological cleaning (opening + erosion) to reduce noise

Phase 3 – Vectorization & Compliance Engine

  - Raster-to-vector conversion of change mask
  - Spatial joins with official buildings and zoning
  - Rule-based compliance classification (zoning conflicts, infrastructure overlap, etc.)
  - Added compactness filter, road distance, and growth type classification (Infill / Edge Expansion / New Settlement)
  - Risk scoring combining rule-based risk, road proximity, and clustering

Phase 4 – Human-in-the-Loop Review Interface

  - Interactive Folium map with risk-based coloring and satellite toggle
  - Click-to-select + auto-zoom functionality
  - Review panel with case status, decision (Valid/Illegal/Uncertain), confidence slider, and time tracking
  - Persistent SQLite logging with Undo capability

Phase 5 – ML Integration & Evaluation

  - RandomForest model trained on engineered features (risk score, area, distances, growth type)
  - "Apply Model Predictions" with live map overlay
  - Evaluation dashboard (accuracy, precision, recall, confusion matrix, reviewer performance)
  - Inspection Priority queue with weighted scoring and class labels

*5. Key Features*

  - Change detection using NDVI built-up proxy
  - Automated compliance rules integrated with official GIS data
  - Interactive map with satellite imagery toggle
  - Human review workflow with confidence and time tracking
  - ML feedback loop with predictions overlay
  - Active Learning mode (uncertainty-based sampling)
  - Priority-based inspection queue for field teams
  - Full audit trail via SQLite


*6. Current Limitations*

  - Sentinel-2 10m resolution limits detection of very small structures
  - Active learning is simulated (manual retraining trigger)
  - Model learns partly from rule-based risk score (some circularity)
  - No automatic background retraining yet
  - Evaluation is based on the same reviewer pool

*7. Future Work*

  - Integrate higher-resolution imagery (Planet Labs or aerial)
  - Implement true asynchronous active learning with auto-retraining
  - Add model versioning, performance gating, and rollback
  - Incorporate spatial diversity sampling
  - Add reviewer reliability scoring
  - Deploy as a shared web application (Docker / Streamlit Community Cloud)
  - Export reports (PDF/CSV) for planning departments
