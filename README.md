# Face Mask Detection System

A robust deep learning system to detect whether individuals are wearing face masks properly or not. This project uses PyTorch and the MobileNetV2 architecture for efficient and accurate mask detection.

## Features

- **Binary Classification**: Detects whether a person is wearing a mask or not
- **Real-time Detection**: Process webcam feed for live mask detection
- **High Accuracy**: Achieves ~99% accuracy on validation data
- **Multi-face Detection**: Can detect and classify multiple faces in the same image/video
- **Status Summary**: Provides an overall status report for all faces detected in an image
- **GPU Acceleration**: Utilizes CUDA for faster training and inference (if available)

## Demo

The system can:
- Process individual images
- Process video files
- Use webcam for real-time mask detection

Each detection includes:
- Bounding box around detected faces
- Color-coded results (green for masked, red for unmasked)
- Confidence score for each prediction
- Overall status for the image/frame

## Requirements

- Python 3.6+
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- tqdm
- Pillow (PIL)

## Installation

```bash
# Create and activate a virtual environment (optional but recommended)
conda create -n maskdetect python=3.8
conda activate maskdetect

# Install dependencies
pip install torch torchvision
pip install opencv-python numpy matplotlib tqdm pillow
```

## Dataset

The model was trained on the Face Mask Detection dataset from Kaggle, containing 12K images of people with and without masks.

The data directory should have the following structure:
```
Face Mask Dataset/
  ├── Train/
  │   ├── WithMask/
  │   └── WithoutMask/
  ├── Validation/
  │   ├── WithMask/
  │   └── WithoutMask/
  └── Test/
        ├── WithMask/
        └── WithoutMask/
```

You can download the dataset from [Kaggle](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset) or use the Kaggle API:

```python
import kagglehub
path = kagglehub.dataset_download("ashishjangra27/face-mask-12k-images-dataset")
```

## Usage

### Training

To train the model:

```bash
python facemask_detector.py --train
```

The best model will be saved as `best_model.pth` in the current directory.

### Demo/Inference

To run the demo:

```bash
python facemask_detector.py --demo
```

You'll be prompted to choose:
1. Process a single image
2. Process a video file
3. Use your webcam for real-time detection
4. Exit

When processing an image, the result will be displayed and saved as `output_[filename]`.

### Key Insights

- The system provides a status summary for each detected face
- The overall status is displayed at the bottom of the processed images/video frames
- For webcam usage, press 'q' to quit

## Model Architecture

The model uses a pre-trained MobileNetV2 as the backbone, with the following modifications:

- The base model's parameters are frozen (transfer learning)
- A custom classifier is added with:
  - Dropout (0.2) for regularization
  - Final linear layer for binary classification

## Performance

With the default settings, the model achieves:
- Training accuracy: ~98.5%
- Validation accuracy: ~99.7%
- Fast inference time suitable for real-time applications

## Future Improvements

Potential enhancements:
- Add a third class for "incorrectly worn masks"
- Implement model quantization for faster inference
- Create a web interface for easy usage
- Add batch processing for multiple images

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.