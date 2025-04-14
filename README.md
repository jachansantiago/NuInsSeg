# NuInsSeg

NuInsSeg is a repository focused on nuclei instance segmentation in H&E-stained histological images. It provides tools and models to facilitate segmentation tasks in computational pathology.

## Getting Started

To utilize the tools and models provided in this repository, follow these steps:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/jachansantiago/NuInsSeg.git
Install Dependencies:

Ensure you have the necessary Python packages installed. You can use the following command:

```bash
pip install -r requirements.txt
```
(Note: The requirements.txt file should list all required packages. If it's not present, please refer to the scripts for package requirements.)

## Prepare the Dataset:

Download the NuInsSeg dataset from Kaggle or Zenodo.

Organize the dataset as expected by the data loading scripts in data.py.

## Train a Model:

To train the SAM model:

```bash
python train_sam.py
```
To train the U-Net model:

```bash
python train_unet.py
```
Adjust configurations in cfg.py as needed for your training setup.



