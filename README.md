# uwmgi

This repo is for Kaggle competition: UW-Madison GI Tract Image Segmentation and for personal final projects on UW CSE455 Computer Vision. 

Kaggle Competition could be found: https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation/overview.

Training curves: https://wandb.ai/alanlee/uw-maddison-gi-tract/runs/39wyhil5?workspace=user-alanlee.

Validation results: 
(80% training, 20% validation, single fold)
- Valid Dice Score: 0.90
- Valid Jaccard Score: 0.87



Introduction video: https://drive.google.com/file/d/1ZEiDCm69xM8BcF9ew87vfs1HxY627rcD/view?usp=sharing.
- Demo: ./pred_demo.ipynb
- Preprcessing and related tools: ./preprocessing.py
- Datasets adn related tools: ./datasets.py
- Main Training scripts and tools: ./train.py
- More details on data analysis, training curves, hyperparameters, etc. in ./summary.pdf: https://github.com/lihaoxin2020/uwmgi/blob/main/sammury.pdf


All references are commented in the code at where the specific snippet has been used. 
