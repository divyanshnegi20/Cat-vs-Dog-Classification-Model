# Cat-vs-Dog-Classification-Model
A complete Deep Learning project implementing a Convolutional Neural Network (CNN) for binary image classification - distinguishing between cats and dogs using TensorFlow/Keras.

PROJECT TITLE: Convolutional Neural Network for Cat vs Dog Image Classification

OBJECTIVE: 
Develop CNN for binary classification of cat/dog images using TensorFlow/Keras.
YOU CAN ACCESS THE DATASET FROM KAGGLE.

MODEL ARCHITECTURE:
Input(256×256×3) → [Conv2D(32)→BN→MaxPool] → [Conv2D(64)→BN→MaxPool] 
→ [Conv2D(128)→BN→MaxPool] → Flatten → Dense(128)→Dropout → Dense(64)→Dropout 
→ Dense(1,Sigmoid)

TECH STACK: TensorFlow/Keras 2.x | Adam Optimizer | Binary Crossentropy
DATA: 256×256 RGB images | train/test directories
FILES: dog_cat_model.keras | train_model.py | evaluate_model.py | architecture.png

QUICK START:
1. pip install tensorflow matplotlib seaborn scikit-learn
2. Place train/test folders with cat/dog subfolders
3. python train_model.py    → dog_cat_model.keras + plots
4. python evaluate_model.py → Full model evaluation
5. python visualize_architecture.py → architecture.png

AUTHOR: Nanu Negi | Dehradun, India
