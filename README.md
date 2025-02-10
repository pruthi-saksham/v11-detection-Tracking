# 🚦 Traffic Analysis using YOLO and DeepSORT

## 📌 Project Overview

This project uses **YOLOv11** for real-time **vehicle detection** and **DeepSORT** for object tracking to analyze traffic flow from video footage. The system identifies vehicles, tracks their movement, detects congestion zones, and generates reports with business insights.

## ⚙️ Installation & Setup

### **1️⃣ Clone the Repository**

```bash
git clone https://github.com/pruthi-saksham/v11-detection-Tracking
```

### **2️⃣ Create and Activate Virtual Environment**

```bash
python -m venv myenv  # Create virtual environment
source myenv/bin/activate  # Mac/Linux
myenv\Scripts\activate  # Windows
```

### **3️⃣ Install Dependencies**

```bash
pip install -r requirements.txt
```

### **4️⃣ Download YOLO Weights**

Ensure the `yolo11n.pt` model is present in the `models/` folder. If not, download it from:

```bash
https://github.com/pruthi-saksham/v11-detection-Tracking/blob/main/yolo11n.pt
```

---

## 🚀 Running the Project

### **1️⃣ Detect & Track Vehicles**

Run the `main.py` script to process a video and generate tracking data.

```bash
python main.py
```

### **2️⃣ Analyze Traffic Data**

Evaluate tracking results and generate insights using `eval.py`:

```bash
python eval.ipynb
```

---

## 📊 Key Features

✅ **Real-time Object Detection:** Uses **YOLOv11** for fast vehicle detection.\
✅ **Multi-Object Tracking:** Utilizes **DeepSORT** to track vehicles across frames.\
✅ **Traffic Flow Analysis:** Identifies congestion zones and movement patterns.\
✅ **Automated Reports:** Generates a CSV file and a business insights report.

---

## 📌 Output Files

📂 **Tracking Data:** `outputs/tracking_data.csv`

```
Frame, Object_ID, Object_Class, Movement_Trend, Zone
148, 2, Car, Right, Left Lane
200, 5, Truck, Left, Right Lane
```

📂 **Business Insights Report:** `BusinessReport.md`

```
Total Vehicles Detected: 320
Heavy Traffic Zones: Intersection A
Peak Congestion Time: Frame 148
```

---

## 📧 Contact

📩 **Saksham Pruthi**\
✉️ **[sakshampruthi9@gmail.com](mailto\:sakshampruthi9@gmail.com)**\
🔗 **https\://github.com/pruthi-saksham**

