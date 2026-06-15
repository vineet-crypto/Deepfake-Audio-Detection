# 🎙️ Deepfake Audio Detection

A deep learning system to classify speech recordings as **Genuine (Human)** or **Deepfake (AI-Generated)**.

---


## 🧠 Methodology

- **Feature Extraction:** Raw `.wav` files are converted into Mel Spectrograms (128 mel bins, 16kHz sample rate, 4s max duration) using `librosa`.
- **Model:** A custom CNN (`DeepfakeAudioDetectorCNN`) with 3 convolutional blocks, batch normalization, dropout, and a sigmoid output for binary classification.
- **Training:** BCELoss + Adam optimizer (lr=0.001), 5 epochs, 80/20 train-validation split.
- **Threshold:** EER-optimal decision threshold is computed at evaluation time instead of a fixed 0.5.

---

## 📊 Results

| Metric | Required | Achieved |
|--------|----------|----------|
| Accuracy | ≥ 80% | 99.37% |
| EER | ≤ 12% | 0.61% |
| F1 Score | ≥ 80% | 99.37% |

---

## 🚀 Setup & Usage

```bash
pip install -r requirements.txt
```

**Train the model:** Run `deepfake_audio_detector.ipynb` end-to-end.

**Test on a new audio file:**
```bash
python inference.py --audio_path /path/to/audio.wav
```

**Run the web app:**
```bash
streamlit run app.py
```

---

## 🌐 Dataset

[The Fake-or-Real Dataset](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset) — `train` folder inside the `LA norm` directory.

---

## 📄 License

Submitted as part of **Mars Open Project 2026** for educational and research purposes.
