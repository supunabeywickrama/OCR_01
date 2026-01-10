# 🚗 AI-Based Intelligent Vehicle  Analytics, License Plate Recognition & Traffic Analytics System

OCR_01 is a **full end-to-end computer vision system** designed for **real-world traffic video analysis**.  
It detects vehicles, recognizes license plates using OCR, preserves plate identity across frames, protects privacy through blurring, and accurately counts vehicles using a **direction-aware virtual line**.

This project is built to work **robustly on any traffic video**, even when:
- License plates are partially visible
- OCR fails intermittently
- Vehicles overlap or move at different speeds

---

## 🎬 Demo Video

<p align="center">
  <a href="https://drive.google.com/file/d/1QD12kpgclO6rrqsKgtlYlex7j13Y7Mqg/view?usp=sharing">
    <img src="https://img.shields.io/badge/▶%20Watch%20Demo--red?style=for-the-badge">
  </a>
</p>

> Replace `YOUR_DEMO_VIDEO_ID` with your demo link  
> (YouTube / Google Drive / GitHub video supported)

---

## 🎯 Project Objectives

The primary goals of this project are:

- Automatically detect **all vehicles** in a video
- Detect and recognize **license plates**
- Maintain **persistent plate identity per vehicle**
- Handle OCR failures gracefully
- Accurately **count vehicles by direction**
- Ensure **privacy compliance** by blurring plates
- Produce **clean, annotated output videos**
- Export structured data for analysis

---

## 🔥 Key Features

- 🚘 **Vehicle Detection** (YOLOv8)
- 🔍 **License Plate Detection** (Custom YOLO model)
- 🧠 **OCR with Sticky Memory**
  - Once a plate is detected, it stays fixed on the vehicle box
  - If OCR fails later → shows `CANT IDENTIFY PLATE`
- 🟥🟩 **Color-coded Plates**
  - Green → Plate successfully recognized
  - Red → Plate unreadable
- 🧮 **Virtual Line Vehicle Counting**
  - Accurate **Inbound / Outbound** counting
  - Direction-aware logic (no double counting)
- 🧱 **Bounding Box Tracking**
  - Plate text stays attached until vehicle disappears
- 🔐 **License Plate Blurring** (Privacy-friendly)
- 📊 **CSV / JSON Export** for analytics
- 🎥 Works on **any video input**

---


## 🔥 About Features (Explained)

### 🚘 Vehicle Detection
- Uses **YOLOv8** for high-speed, high-accuracy vehicle detection
- Supports cars, buses, trucks, bikes (depending on model)
- Works reliably under different lighting and camera angles

---

### 🔍 License Plate Detection
- Uses a **custom-trained YOLO model (`best.pt`)**
- Detects license plate regions inside vehicle bounding boxes
- Reduces false positives by vehicle-aware cropping

---

### 🧠 OCR with Sticky Memory (Key Innovation)

OCR in real-world videos is unstable.  
To solve this, OCR_01 introduces **Sticky Plate Memory**:

- When a plate is recognized once → it is **stored permanently** for that vehicle
- If OCR fails in later frames → the last valid plate is reused
- If OCR never succeeds → label shows  
            plate: CANT IDENTIFY PLATE

✔ Plate text remains fixed on the vehicle  
✔ Prevents flickering and text changes  
✔ Greatly improves usability

---

### 🎨 Smart Color-Coded Plate Labels

- 🟢 **Green text** → Plate successfully recognized
- 🔴 **Red text** → Plate could not be identified

This allows instant visual understanding of OCR quality.

---

### 🧱 Vehicle Tracking (IoU-Based)

- Lightweight IoU tracker
- Assigns a **unique ID** per vehicle
- Tracks vehicles across frames
- Automatically removes stale tracks

Tracking enables:
- Persistent plate labels
- Accurate counting
- Direction detection

---

### 📏 Virtual Line Vehicle Counting

OCR_01 uses a **virtual line** to count vehicles:

- User defines 2 coordinate points
- Vehicles are counted **only when crossing the line**
- Direction is determined using previous and current positions

#### Direction Support:
- **Inbound**
- **Outbound**

✔ No double counting  
✔ Works even with slow or fast vehicles  
✔ Reliable in dense traffic

---

### 🔐 License Plate Blurring (Privacy)

- Detected plates are blurred automatically
- Ensures compliance with privacy regulations
- Blurring does not affect OCR or tracking

---

### 📊 Data Export

Automatically exports structured data:

- CSV
- JSON

Each record includes:
- Vehicle ID
- License plate number
- Direction
- Timestamp / frame index

Perfect for:
- Traffic analytics
- Research
- Dashboards
- Reports

---

## 🖼️ Output Visualization

Each processed video includes:

- Vehicle bounding boxes
- License plate bounding boxes
- Fixed plate label (non-flickering)
- Virtual counting line
- Inbound / Outbound counters
- Color-coded plate text

Output saved as:
      outputs/annotated.mp4
---

## 📁 Project Structure (Detailed)

<img width="559" height="659" alt="image" src="https://github.com/user-attachments/assets/ed41e91d-a5f5-447a-8e47-84040d8a9ee3" />
      
---


## 🛠 Technologies Used

Python 3.10
Ultralytics YOLOv8
OpenCV
Supervision
Tesseract OCR
NumPy

---

## 🚀 Applications

Smart traffic monitoring
Intelligent transportation systems (ITS)
Parking automation
Toll booth automation
Surveillance & analytics
Academic research projects

---

## 👤 Author

- Supun Abeywickrama
- AI / ML | Computer Vision | Robotics
 

---
