# Animal detection with Faster R-CNN: Transfer Learning and Domain Adaptation
A Multi-domain generalizable animal detection model: exploration of potential solutions to improve the SOA. 


## Table of content

- [Requirements](#requirements)
- [Getting started](#getting-started)
- [Examples of experiments](#examples-of-experiments)
- [Authors](#authors)

## Requirements

- Python 3.7.11
- An ipython notebook editor and runner (Jupyter Notebook, VSCode with extension, etc...)
- Having read our report

## Getting started

1. Download the Git repo using:

    ```
    git clone https://github.com/Hazot/Animal-Detection-and-Generalization.git
    ```
   or download it manually.


2. Download the dataset (Benchmark images: 6GB; Metadata files: 3MB) under the header CCT20 Benchmark subset: https://lila.science/datasets/caltech-camera-traps. Extract the dataset directly into the project folder. Make sure to have the following folders:
- eccv_18_all_images_sm
- eccv_18_annotation_files
4. Load the virtual environment librairies using:

    ```
    python -m pip install -r requirements.txt
    ```
5. Open the main_notebook.ipynb and simply execute the cells sequentially.

## Examples of experiments

1. To reproduce the test results of a RPN+ROI model, simply execute the cells until the interactive part.
Make sure that you use the right parameters: 
    ```
    data_augmentation_mode = 'none'
    model_depth = 3
    ```
    In the cell below:
    ```
    lightweight_mode = 1
    ```
    This will use smaller datasets and only 5 epochs. Exercute every cells until right before the 
    optional part (Make Predictions with a model). The results if this training will not be useful at all.
    To get better results, you need a lot of time (4-5 hours) and disable the ligthweight_mode by doing:
    ```
    lightweight_mode = 0
    ```
2. To use domain adaptation on this model, after training a model go to the Domain Adaptation heading and 
run every cell below until the very last.


## Authors

- Abdiel Fernandez (abdielfer@gmail.com)
- Rose Guay Hottin (guayhottin.rose@hotmail.com)
- Kevin Lessard (kevin.lessard@umontreal.ca)
- Santino Nanini (santino.nanini@umontreal.ca)
