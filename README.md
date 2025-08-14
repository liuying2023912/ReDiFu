# ReDiFu
Relational graph-driven differential denoising and diffusion fusion for multimodal conversation emotion recognition
## Setup
### 1. Install Dependencies
Install all required packages using the following command:
```bash
pip install -r requirements.txt
```
### 2. Download Preprocessed Datasets
Download the preprocessed datasets from [here](https://drive.google.com/drive/folders/1mzyt_zUV4d4DuGwUCx4aSyF77msnF2-Q?usp=sharing), and put them into `data/`.

## Run Our Model
### 1. Run the model on IEMOCAP dataset:
```bash
python train.py --lr=0.0001 --dropout=0.5 --l2=0.00005 --batch-size=16 --hidden_dim=512 --n_head=64 --windows=20 --epochs=50 --seed=2094 --Dataset="IEMOCAP" --save_model_path="./IEMOCAP"
```
### 2. Run the model on MELD dataset:
```bash
python train.py --lr=0.00005 --dropout=0.6 --l2=0.00005 --batch-size=16 --hidden_dim=256 --n_head=16 --windows=25 --epochs=20 --seed=123 --Dataset="MELD" --save_model_path="./MELD"
```
