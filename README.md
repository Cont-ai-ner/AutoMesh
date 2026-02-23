# AutoMesh: A Generative AI System for 3D Object Reconstruction from Single or Multiple Images
AutoMesh is an automated pipeline designed to convert a single 2D RGB photograph into a physics-ready, textured, and walkable 3D room environment suitable for Unity 3D and VR platforms.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## **🛠️ Tech Stack**

* **Depth Estimation:** ZoeDepth (Metric-scale depth prediction)
* **Object Segmentation:** SAM-2 (Segment Anything Model 2)
* **3D Reconstruction:** TripoSR (Fast Transformer-based feed-forward reconstruction)
* **Scene Integration:** Trimesh (Mesh processing and scene assembly)
* **Interface:** Gradio (Web-based UI for model interaction)

---

## **📂 Repository Structure**

```text
AutoMesh/
├── run_automesh.py      # Main Pipeline Controller & UI
├── libs/                # External Library Repositories (Placeholders)
│   ├── ZoeDepth/        # Metric Depth library
│   ├── TripoSR/         # 3D Reconstruction library
│   └── sam2/            # Segment Anything 2 library
├── weights/             # Model Checkpoints (Place .pt / .ckpt files here)
├── output/              # Final exported .glb assets and debug logs
└── README.md

```

---

## **🚀 Setup Instructions**

### **1. Clone External Libraries**

Navigate to the `libs/` directory and clone the required repositories:

```bash
git clone https://github.com/isl-org/ZoeDepth.git libs/ZoeDepth

git clone https://github.com/stabilityai/TripoSR.git libs/TripoSR

git clone https://github.com/facebookresearch/segment-anything-2.git libs/sam2
cd libs/sam2 && pip install -e .

```

### **2. Download Model Weights**

Download the following pre-trained models and place them in the `weights/` folder:

* **ZoeDepth:** `ZoeD_M12_N.pt` (Place in `weights/` or `checkpoints/`)
* **SAM-2:** `sam2.1_hiera_large.pt`
* **TripoSR:** `model.ckpt` (and associated `config.yaml`)

### **3. Install Dependencies**

Ensure you have an active Python 3.10+ environment with CUDA support:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install gradio numpy opencv-python trimesh rembg pillow

```

### **4. Usage**

Run the main controller to launch the Gradio web interface:

```bash
python run_automesh.py

```

---

## **📊 Project Status**

* **Phase I:** Feasibility study and architectural design (Completed).
* **Phase II (MVP):** Single-image reconstruction, metric depth calibration, and Unity integration (Completed).
* **Phase II (Expansion):** 360-degree scene outpainting and high-fidelity texture mapping (In Progress).

---

## **⚖️ License**

This project incorporates several open-source models. Please refer to the individual licenses of **ZoeDepth (MIT)**, **SAM-2 (Apache 2.0)**, and **TripoSR (MIT)** for usage details.

---
