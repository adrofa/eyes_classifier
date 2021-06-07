# Task Description
Build binary classifier for dataset consisting of grayscale images of size 24x24 pixels.

* Positive class - opened eyes:
  * sample: ![pos](./notebooks/pos_example.png)
* Negative class - closed eyes:
  * sample: ![neg](./notebooks/neg_example.png)
  
**Note:** data is not annotated.

# Outputs

### OpenEyesClassificator Class
`OpenEyesClassificator` is here: [`./open_eyes_classificator/main.py`](./open_eyes_classificator/main.py)

You can check it via this notebook
*(it is independent to Jupyter startup folder, but <u>keep the project structure as it is</u>)*:
[`./notebooks/open_eyes_classificator_demo.ipynb`](./notebooks/open_eyes_classificator_demo.ipynb)

`OpenEyesClassificator.__init__` methods contains 2 parameters:
* `model_type` (default: `ensemble`):
  * `ensemble`  - use ensemble of 5 models for prediction;
  * `single` - use single model for prediction.
* `device` (default: `cpu`): to run on CPU or GPU.

#### Requirements
* [`requirements.txt`](requirements.txt) - contains the short list of required packages
  (other packages are in the dependencies of the listed ones);
* [`requirements_full.txt`](requirements_full.txt) - contains the full list of required packages
  (produced by `pip freeze` in the project's venv.
  
#### Report
Report is below.

# Report

## Classifier Performance
**My final classifier (ensemble of 5 models) demonstrated 0.9775 accuracy on the hidden dataset,
which corresponds to 0.0691 log-loss.**

*My threshold for binary prediction is 0.59. I selected it on cross-validation:*
* wide-selection:
  * ![wide](./notebooks/inference_analysis/preview/accuracy_ths_selection.png)
* narrow-selection
  * ![narrow](./notebooks/inference_analysis/preview/accuracy_ths_selection_2.png)
  
*Images are produced by this notebook:
[`./notebooks/inference_analysis/accuracy_threshold_selection.ipynb`](./notebooks/inference_analysis/accuracy_threshold_selection.ipynb).*
  
Worth to mention, that a lot of false classifications (especially FN)
are dedicated to images, with not clear pictures
(which is unclear even for a human - I mean myself).   

### False Positive:
![FP](./notebooks/inference_analysis/preview/fp.png)

### False Negative
![FN](./notebooks/inference_analysis/preview/fn.png)

### True Positive wih Low Confidence (<0.7)
*all TP predictions are with a confidence >0.7*

### True Negative wih Low Confidence (>0.3)
![TP](./notebooks/inference_analysis/preview/tn_low_conf.png)

*Images are produced by this notebook:
[`./notebooks/inference_analysis/final_model_performance.ipynb`](./notebooks/inference_analysis/final_model_performance.ipynb).*

## Data Annotation
My first thought was to annotate ~500 images manually and annotate additional data if needed.
Then I decided to google something about opened-closed eyes classification problem and found
[**Closed Eyes In The Wild (CEW) dataset**](http://parnec.nuaa.edu.cn/_upload/tpl/02/db/731/template731/pages/xtan/ClosedEyeDatabases.html).

Eye patches from CEW dataset looked very similar to the original data.
I thought that it might be not the similar one, but the same.

I checked and found that the datasets are identical, except 1 thing:
the original dataset lacks 850 images.

*P.S. `cv2.imread` returns `None` for 4 of the mentioned 850 images from CEW dataset.*

*For more details re datasets' identity check, please, consider:
[`./utils/crossval_split.py`](./utils/crossval_split.py)*.

**CEW dataset solved the annotation "problem".**


## Test-Train-Valid- Split
For models training and validation I used CWE dataset. I split the data in the following way: 
* **test / hidden**: 846 from CEW dataset, which do not exist in the original data.
  I used this images only when the final model was ready to perform the final check;
* **train / valid**: 5-folds cross-validation -> 3200 images for training and 800 images for validation

Test-Train-Valid split script: [`./utils/crossval_split.py`](./utils/crossval_split.py)


## Model Selection
Usually I start my research from pretrained SOTA models, but:
* SOTA architectures are too complicated and, most likely, will overfit on such simple data;
* current SOTA architectures are pretrained on data from different distribution.

Hence, I decided to compile and fit my custom model.
My research process is described below in section ["Research Process"](#research-process) 

*BTW, if you'd like to see how I utilize transfer learning, please, consider
[this repository](https://github.com/adrofa/cats-dogs-detection).*


## Repository Structure
### [`./dat`](./data)
* [`/dataset_B_Eye_Images`](./data/dataset_B_Eye_Images) -
  [CEW dataset](http://parnec.nuaa.edu.cn/_upload/tpl/02/db/731/template731/pages/xtan/ClosedEyeDatabases.html);
  

* [`/EyesDataset`](./data/EyesDataset) - original dataset.

### [`./notebooks`](./notebooks)
* [`/inference_analysis`](./notebooks/inference_analysis) -
different notebooks for results analysis and their outputs (*.png files);
  

* [`/lr_finder`](./notebooks/lr_finder) - notebooks for learning rate selection via
  [`torch-lr-finder` package](https://github.com/davidtvs/pytorch-lr-finder) 
  * `/vN.ipynb` - N corresponds to Hypothesis number in the ["Research Process"](#research-process) section.
  
### [`./open_eyes_classificator`](./open_eyes_classificator)
* [`main.py`](./open_eyes_classificator/main.py) contains `OpenEyesClassificator` class.

### [`./output`](./output)

* [`/crossval_split`](./output/crossval_split):
  * [`/identity_df.pkl`](./output/crossval_split/identity_df.pkl) -
    pickle-dump of a `pandas.DataFrame` with columns:
      * `cew_img` - path to an image from [CEW dataset](./data/dataset_B_Eye_Images);
      * `original_img` - path to an image from [the original dataset](./data/EyesDataset);
      * `label` - image class: 1 - opened, 0 - closed; 
  * [/crossval_dct.pkl](./output/crossval_split/crossval_dct.pkl)
    pickle-dump with a dict with keys:
      * `hidden` - part of the `identity_df` with `test / hidden` images;
      * `1` - `5` - folds with train and valid parts of the `identity_df`;
  

* [`/image_normalization`](./output/image_normalization):
  * [`/normalization_params.txt`](./output/image_normalization/normalization_params.txt) -
    mean and std of CEW dataset;

    
* [`/models`](./output/models):
  * `/hypothesis-1` - `/hypothesis-7` - outputs (including models' weights) of Hypothesis
    described in ["Reseacrh Process" section](#research-process);
    * `/fold-1` - `/fold-5` - folds of a corresponding hypothesis
      *(all folds exist only for Hypothesis 4)*.
      
### [`./train`](./train)
* [`/versions`](./train/versions) - versions of the training pipeline parts:
  * [`/augmentation.py`](./train/versions/augmentation.py) - augmentation's versions;
  * [`/criterion.py`](./train/versions/criterion.py) - loss-function's versions;
  * [`/model.py`](./train/versions/model.py) - model's versions (including models' classes);
  * [`/optimizer.py`](./train/versions/optimizer.py) - optimizer's versions;
  * [`/scheduler.py`](./train/versions/scheduler.py) - scheduler's versions.
  
### [`./utils`](./utils)
* [`/crossval_split.py`](./utils/crossval_split.py) - script for
  (1) identity of the original and CEW datasets check; (2) cross-validation split;
  
* [`/utils/image_normalization.py`](./utils/image_normalization.py) - script for collecting
  CEW dataset's stats (mean and std) for image_normalization;
  
* [`/support.py`](./utils/support.py) - contains different support functions.


## Research Process
Before fitting the model I estimated dataset's mean and std for normalization
(code is here: [`./utils/image_normalization.py`](./utils/image_normalization.py)).

Details re versions of:
* models: [`./train/versions/model.py`](./train/versions/model.py)
* loss-functions: [`./train/versions/criterions.py`](./train/versions/criterion.py)
* augmentation: [`./train/versions/augmentation.py`](./train/versions/augmentation.py)
* optimizers: [`./train/versions/optimizer.py`](./train/versions/optimizer.py)
* schedulers: [`./train/versions/scheduler.py`](./train/versions/scheduler.py)

*Loss and accuracy charts below are for the fold-1.*

#### Hypothesis 1
In this hypothesis I compile a basic CNN, which consists of 2xConv + 3xFC layers

Config-file: [`./output/models/hypothesis-1/fold-1/config.json`](./output/models/hypothesis-1/fold-1/config.json)

* **`Model version: 1`**
* **`Optimizer version: adam_1`** (lr: 3.20E-02)
  * lr found via torch_lr_finder: [`./notebooks/lr_finder/v1.ipynb`](./notebooks/lr_finder/v1.ipynb)
* **`Augmentation version: 1`** (Normalization + Horizontal flip + Brightness + Contrast + Blur)
* **`Scheduler version: rop_1`** (reduce-on-plateau after 5 epochs)
* **`Criterion version: 1`** (Binary Cross Entropy)

![hypothesis-1](./output/models/hypothesis-1/fold-1/progress.png)

During several last epochs both of the losses have not been updating:
it looks like gradients exploding.

#### Hypothesis 2
To overcome gradients exploding I will add BatchNorm layer before the second Conv layer.

Config-file: [`./output/models/hypothesis-2/fold-1/config.json`](./output/models/hypothesis-2/fold-1/config.json)

* **`Model version: 2`**
* **`Optimizer version: adam_2`** (lr: 4.53E-03)
  * lr found via torch_lr_finder: [`./notebooks/lr_finder/v2.ipynb`](./notebooks/lr_finder/v2.ipynb)
* **`Augmentation version: 1`**
* **`Scheduler version: rop_1`**
* **`Criterion version: 1`**

![hypothesis-2](./output/models/hypothesis-2/fold-1/progress.png)

#### Hypothesis 3
BatchNorm helped. Model doesn't reach 100% accuracy on the training set =>
I will add 1 Conv and 1 FC layers.

*BTW, I am increasing the model size in 3 directions: depth, width, resolution
(EfficientNet paper showed, that increasing of 3 directions simultaneously is more reasonable).*

Config-file: [`./output/models/hypothesis-3/fold-1/config.json`](./output/models/hypothesis-3/fold-1/config.json)

* **`Model version: 3`**
* **`Optimizer version: adam_3`** (lr: 3.43E-03)
  * lr found via torch_lr_finder: [`./notebooks/lr_finder/v3.ipynb`](./notebooks/lr_finder/v3.ipynb)
* **`Augmentation version: 1`**
* **`Scheduler version: rop_1`**
* **`Criterion version: 1`**

![hypothesis-3](./output/models/hypothesis-3/fold-1/progress.png)

Adding additional Conv and FC layers improved metrics.

#### Hypothesis 4
Let's add some layers.

Config-file: [`./output/models/hypothesis-4/fold-1/config.json`](./output/models/hypothesis-4/fold-1/config.json)

* **`Model version: 4`**
* **`Optimizer version: adam_4`** (lr: 1.79E-03)
  * lr found via torch_lr_finder: [`./notebooks/lr_finder/v4.ipynb`](./notebooks/lr_finder/v4.ipynb)
* **`Augmentation version: 1`**
* **`Scheduler version: rop_1`**
* **`Criterion version: 1`**

![hypothesis-4](./output/models/hypothesis-4/fold-1/progress.png)

At this step I decided to look at the model inference in more details. Takeaways:
* there are some mistakes in data annotation;
* some images can be hardly classified by human (me).
<details>
  <summary><b>Images (Valid Dataset)</b></summary>

##### False Positive
![fp](./notebooks/inference_analysis/preview/v4/fp.png)
##### False Negative
![fn](./notebooks/inference_analysis/preview/v4/fn.png)
##### True Positive with Low Confidence (<0.7)
![tp_lc](./notebooks/inference_analysis/preview/v4/tp_lowConf.png)
##### True Negative with Low Confidence (>0.3)
![tn_lc](./notebooks/inference_analysis/preview/v4/tn_lowConf.png)
</details>

#### Hypothesis 5
In Hypothesis 4 the model overfitted (last epoch losses: train-0.021; valid-0.101) .
I will try to overcome overfitting by improving augmentation.

Also, I noticed that the valid-loss starts to decrease after reduce-lr-on-plateu,
<u>even if it was rising for several epochs before</u>.
Hence, I will reduce `patience` parameter in the LR-scheduler.

Config-file: [`./output/models/hypothesis-5/fold-1/config.json`](./output/models/hypothesis-5/fold-1/config.json)

* **`Model version: 4`**
* **`Optimizer version: adam_4`**
* **`Augmentation version: 2`**
* **`Scheduler version: rop_2`**
* **`Criterion version: 1`**

![hypothesis-5](./output/models/hypothesis-5/fold-1/progress.png)

#### Hypothesis 6
Additional augmentation helped to slightly decrease valid-loss In Hypothesis 5.
In this Hypothesis I will try to add dropout layers to FC layers.

Config-file: [`./output/models/hypothesis-6/fold-1/config.json`](./output/models/hypothesis-6/fold-1/config.json)

* **`Model version: 5`**
* **`Optimizer version: adam_5`** (lr: 2.36E-03)
  * lr found via torch_lr_finder: [`./notebooks/lr_finder/v5.ipynb`](./notebooks/lr_finder/v5.ipynb)
* **`Augmentation version: 2`**
* **`Scheduler version: rop_2`**
* **`Criterion version: 1`**

![hypothesis-6](./output/models/hypothesis-6/fold-1/progress.png)

Dropout layers in helped to overcome overfitting, but the  valid-loss increased.

#### Hypothesis 7
In this Hypothesis I will try to increase FC layers width (to provide more data to layers with dropout).

Config-file: [`./output/models/hypothesis-7/fold-1/config.json`](./output/models/hypothesis-7/fold-1/config.json)

* **`Model version: 6`**
* **`Optimizer version: adam_6`** (lr: 2.15E-03)
  * lr found via torch_lr_finder: [`./notebooks/lr_finder/v6.ipynb`](./notebooks/lr_finder/v6.ipynb)
* **`Augmentation version: 2`**
* **`Scheduler version: rop_2`**
* **`Criterion version: 1`**

![hypothesis-7](./output/models/hypothesis-7/fold-1/progress.png)

Increasing models' width didn't help.

### Step 6: Fit the Model Selection
Finally, I decided to stick to the model from Hypothesis 5 and cross-validate it on the other folds.


    



# DRAFTS ---------------------------
I will try to use a ready-to-go architecture to save time on writing a model from scratch.
If it fails, I will write my custom model.

I considered [Papers with Code](https://paperswithcode.com/)
section ['Image Classification on ImageNet'](https://paperswithcode.com/sota/image-classification-on-imagenet)
to select one of the recent well-established architectures.

For my first experiments I selected ["EfficientNet-B0"](https://arxiv.org/abs/2003.08237).
This architecture is relatively small - 5.3m parameters and
precise in [`timm` package](https://rwightman.github.io/pytorch-image-models/) with weights.  


This approach will allow:
* to reach 100% accuracy in case the hidden dataset consists of missed 850 images,
  because [n_folds - 1] of models will be fitted on this data;
* to reach accuracy comparable with mean cross-validation accuracy in case hidden dataset consists of 
  another images, but from the same distribution as images I have;
* also, using an ensemble of models usually provides better results then a single model.
  Inasmuch I do not have any inference time limitations,
  out-of-fold predictions should be a better solution in any case.
  
; from CEW dataset, which do not exist in the origginal dataset on training;
* validate model on  several models which generalize well (w/o overfitting) on different folds;
* average out-of-fold predictions in `OpenEyesClassificator.predict` method.


### Step 1: Familiarizing preview(pos, 10, 3, "pos_example.png") with the Data
Firstly, I looked at ~300 images to decide what to do next.
I noticed that both classes have strong patterns: e.g., most of closed eyes
images have horizontal line in the center of an image. It shouldn't be hard to classify.
Also, there is no class imbalance. 

**The main problem was the lack of annotation.**

### Step 2: Annotation
I supposed that I could:
* annotate manually ~500 images (and use ~400 for training and ~100 for validation);
* .

#### Datasets' Identity Check
* The data is very small, hence I can load both (original and CEW) datasets into RAM.
* Then I need to compare each image from one dataset with each image from another 
  until the identical image is found.
* To speed up this process, I decided to split images into groups by file size. 
  * 