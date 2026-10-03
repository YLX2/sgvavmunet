# sgvavmunet
This repository provides the implementation, training and evaluation code of Topano-Net
## ✨ Highlights

- 🫀 Designed for coronary artery segmentation in X-ray coronary angiography.
- 🔍 Topology-guided dual-branch architecture for preserving coronary vessel continuity.
- 📐 Anisotropic feature modeling for slender and tortuous vessel structures.
- 🌐 Semantic recomposition and guidance for multi-scale topology-aware decoding.
- 📊 Supports quantitative evaluation and comparison with state-of-the-art segmentation methods.

🔧 Installation
1. Clone the repository
2. Create the environment

conda create -n topano python=3.9
conda activate topano

4. Install dependencies
pip install -r requirements.txt
The main dependencies include:
MONAI
SimpleITK
nibabel
OpenCV
scikit-image
scipy
scikit-learn
timm
MedPy
thop
Additional dependencies are provided in requirements.txt.

📊 Dataset

Experiments are conducted on:

ARCADE public coronary angiography dataset
Clinical XCA dataset containing 170 patients

Dataset Organization

datasets/

├── ARCADE/

│     ├── images/

│     └── masks/

└── Clinical/

      ├── images/
      
      └── masks/
    
Please follow the corresponding dataset licenses and usage policies when downloading and using the datasets.

🚀 Training

To train Topano-Net:

python train.py \
    --dataset ARCADE \
    --model Topano-Net

You can modify the training configuration according to your dataset and hardware environment.

🧪 Testing

Run inference using:

python test.py \
    --dataset ARCADE \
    --model Topano-Net \
    --checkpoint path/to/checkpoint.pth

The predicted segmentation masks will be saved to:

results/

🙏 Acknowledgements

We thank the authors of the related open-source projects and public datasets that contributed to this work.
