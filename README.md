# 👤 Face Detection & Classification with VGG16

This project uses **Transfer Learning (VGG16)** to detect and classify faces from images.  
It applies **data augmentation** with Keras’ `ImageDataGenerator` to improve model generalization.

##  Model Pipeline
1. **Data Preprocessing**
   - Training and testing images loaded from `Datasets/Train` and `Datasets/Test`
   - Applied rescaling, shear, zoom, and horizontal flip for augmentation  

2. **Transfer Learning with VGG16**
   - Pre-trained VGG16 (without top layers) used as feature extractor  
   - Added custom dense layers for classification  

3. **Training**
   - Trained on custom dataset with categorical cross-entropy  
   - Uses `Adam` optimizer and accuracy metric  

## 🛠️ Tech Stack
- Python  
- TensorFlow / Keras  
- NumPy, Matplotlib  
