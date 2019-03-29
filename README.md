# Simple Image Classier using CNN

# How to use this

1. Delete all .gitkeep files
2. Add images to correct/incorrect directory
3. Exec .py script

# Requirements

- keras
- numpy
- cv2

# How to exec

## Training

```
python image_classifier.py train
```

## Validation

```
python image_classifier.py validate
```

# Directory structure

```
.
├── README.md
├── dataset
│   ├── correct
│   │   ├── a.png
│   │   ├── ...
│   │   └── b.png
│   ├── incorrect
│   │   ├── a.png
│   │   ├── ...
│   │   └── b.png
│   └── model.h5
└── image_classifier.py
```
