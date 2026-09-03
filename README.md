# Yoga Posture Monitoring

**Developed by Charan**

A real-time yoga posture monitoring web application that uses your webcam to detect and classify yoga poses using AI-powered pose estimation.

---

## About the Project

This project combines a React-based frontend with a deep learning classification model to provide real-time yoga pose feedback. The system captures live webcam video, estimates body keypoints using Google's MoveNet model, and classifies the detected pose into one of 8 supported yoga postures — all running directly in the browser without any server-side inference.

---

## Features

- Real-time pose detection via webcam
- Supports 8 yoga pose classes: Chair, Cobra, Downward Dog, Shoulder Stand, Triangle, Tree, Warrior, and No Pose
- Instant feedback on pose accuracy
- Tutorial pages with pose guides
- Responsive UI with pose-by-pose instructions

---

## Tech Stack

### Frontend
| Technology | Purpose |
|---|---|
| React 17 | UI framework |
| TensorFlow.js | In-browser model inference |
| @tensorflow-models/pose-detection | PoseNet / MoveNet keypoint detection |
| React Webcam | Live webcam feed |
| React Router DOM v6 | Client-side routing |

### Classification Model (Python)
| Technology | Purpose |
|---|---|
| TensorFlow / Keras | Model training |
| TensorFlow Lite (MoveNet Thunder) | Keypoint extraction from images |
| TensorFlow.js Converter | Export model for browser inference |
| scikit-learn | Train/test split |
| OpenCV | Image preprocessing |
| pandas | CSV data handling |

---

## Model Architecture

The classification model takes **34 normalized landmark coordinates** (17 keypoints × x,y) as input and classifies them into one of 8 pose categories.

```
Input (34)
  → Dense(128, relu6)
  → Dropout(0.5)
  → Dense(64, relu6)
  → Dropout(0.5)
  → Dense(8, softmax)
```

- **Optimizer:** Adam
- **Loss:** Categorical Cross-Entropy
- **Training:** Up to 200 epochs with early stopping on validation accuracy
- **Best weights saved** via ModelCheckpoint callback

---

## Pose Classes

| # | Pose |
|---|---|
| 1 | Chair |
| 2 | Cobra |
| 3 | Downward Dog |
| 4 | No Pose |
| 5 | Shoulder Stand |
| 6 | Triangle |
| 7 | Tree |
| 8 | Warrior |

---

## Project Structure

```
YogaIntelliJ/
├── frontend/                   # React web application
│   ├── public/
│   ├── src/
│   │   ├── components/         # DropDown, Instructions, PoseStart
│   │   ├── pages/              # Home, About, Tutorials, Yoga
│   │   └── utils/              # Data, helpers, music, pose images
│   └── package.json
│
└── classification model/       # Python ML pipeline
    ├── movenet.py              # MoveNet TFLite wrapper
    ├── data.py                 # BodyPart definitions & data types
    ├── proprocessing.py        # Keypoint extraction from image dataset
    ├── training.py             # Model training & export to TF.js
    ├── model/                  # Exported TF.js model (JSON + weights)
    ├── csv_per_pose/           # Per-pose landmark CSV files
    ├── train_data.csv          # Combined training data
    └── test_data.csv           # Combined test data
```

---

## Getting Started

### Frontend

```bash
cd frontend
npm install
npm start
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Classification Model (Python)

```bash
cd "classification model"
pip install tensorflow tensorflow-hub tensorflowjs scikit-learn opencv-python pandas
python training.py
```

The trained model will be exported to the `model/` directory in TensorFlow.js format, ready to be served by the frontend.

---

## How It Works

1. **Data Collection** — Yoga pose images are processed through MoveNet Thunder (TFLite) to extract 17 body keypoints per image.
2. **Preprocessing** — Keypoints are normalized (centered at hip midpoint, scaled by pose size) and flattened into a 34-dimensional embedding.
3. **Training** — A dense neural network is trained on these embeddings to classify poses.
4. **Export** — The trained model is converted to TensorFlow.js format for browser inference.
5. **Real-time Detection** — The React app captures webcam frames, runs MoveNet in the browser to get keypoints, feeds them into the trained classifier, and displays the predicted pose.

---

## Developer

**Charan**
GitHub: [https://github.com/charan6241](https://github.com/charan6241)
