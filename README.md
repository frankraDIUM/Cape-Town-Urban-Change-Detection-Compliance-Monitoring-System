# 🏙️ Cape Town Urban Change Detection Compliance Monitoring System

Research title:
GeoAI-Driven Urban Change Detection and Compliance Monitoring: Integrating Sentinel-2 Imagery, GIS-Based Rules, and Human-in-the-Loop Machine Learning in Khayelitsha, Cape Town

A Human-in-the-Loop GeoAI tool for detecting unauthorized urban development in Khayelitsha, Cape Town. This project combines Sentinel-2 satellite imagery, 
official Cape Town GIS layers, change detection, rule-based compliance, and machine learning to help urban planners identify and prioritize potential land-use violations.

Focus Area: Khayelitsha (Informal Settlement Expansion)


---
Final Preview

<p align="center">
  <img src="https://github.com/frankraDIUM/Cape-Town-Urban-Change-Detection-Compliance-Monitoring-System/blob/main/cape_final.gif" />
</p>

---

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

---

ML Lab
<p align="center">
  <img src="https://github.com/frankraDIUM/Cape-Town-Urban-Change-Detection-Compliance-Monitoring-System/blob/main/urban2.png" />
</p>

---

Evaluation
<p align="center">
  <img src="https://github.com/frankraDIUM/Cape-Town-Urban-Change-Detection-Compliance-Monitoring-System/blob/main/urban3.png" />
</p>

---

Inspection Priority
<p align="center">
  <img src="https://github.com/frankraDIUM/Cape-Town-Urban-Change-Detection-Compliance-Monitoring-System/blob/main/urban4.png" />
</p>

---


*1. Project Overview*

This system combines Sentinel-2 satellite imagery, official City of Cape Town GIS data, change detection, 
rule-based compliance, and machine learning to support urban planning and enforcement teams.


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


*4. Key Features*

  - Change detection using NDVI built-up proxy
  - Automated compliance rules integrated with official GIS data
  - Interactive map with satellite imagery toggle
  - Human review workflow with confidence and time tracking
  - ML feedback loop with predictions overlay
  - Active Learning mode (uncertainty-based sampling)
  - Priority-based inspection queue for field teams
  - Full audit trail via SQLite

