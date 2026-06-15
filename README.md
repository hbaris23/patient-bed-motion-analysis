# Patient Bed Motion Analysis

This project analyzes patient movements in bed using computer vision and deep learning techniques. It uses OpenCV and PyTorch to detect body keypoints, calculate motion scores, detect position changes, and generate analysis outputs.

## Project Purpose

The aim of this project is to track and analyze patient movements on a bed from a video. The system detects upper body keypoints, calculates motion intensity, identifies position changes, and creates output files such as video, graph, CSV log, dashboard image, and text report.

## Technologies Used

* Python
* OpenCV
* PyTorch
* Torchvision
* NumPy
* Matplotlib

## Installation

Before running the project, create a virtual environment and install the required libraries:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Run the Project

After installing the required libraries, run the Python file:

```powershell
python patient-bed-motion-analysis.py
```

## Input Video

The input video must be in the same folder as the Python file.

Required video file:

```text
7556227-uhd_3840_2160_25fps.mp4
```

## Outputs

Example output files are included in the `outputs` folder. After running the project, the following files can also be generated automatically:

- output_fast.mp4
- motion_log.csv
- motion_graph.png
- dashboard.png
- analiz_raporu.txt



## Notes

If CUDA is not available, the model runs on CPU.

The first run may take longer because the PyTorch model weights may be downloaded automatically.
