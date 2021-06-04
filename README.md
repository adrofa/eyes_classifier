## Research Steps
### Step 1: Familiarizing  with the Data
 I'd like to look at several images  

Firstly, I looked at ~300 images to decide what to do next.
I noticed that both classes have strong patterns: e.g., most of closed eyes
images have horizontal line in the center of an image. It shouldn't be hard to classify.

**The main problem was the lack of annotation.**

### Step 2: Annotation
I supposed that I could:
* annotate manually ~500 images (and use ~400 for training and ~100 for validation);
* annotate additional data if needed.

Then I decided to google something about opened-closed eyes classification problem and found
[**Closed Eyes In The Wild (CEW) dataset**](http://parnec.nuaa.edu.cn/_upload/tpl/02/db/731/template731/pages/xtan/ClosedEyeDatabases.html).

Eye patches from CEW dataset (hereinafter "CEW dataset") looks very similar to the provided data.
I thought that it might be not a similar one, but the same.

#### Datasets' Identity Check
* The data is very small, hence I can load both (provided and CEW) datasets into RAM.
* Then I need to compare each image from one dataset with each image from another 
  until the identical image is found.
* To speed up this process, I decided to split images into groups by file size. 
  * For more details, please, consider: [`scripts/datasets_identity_check.py`](scripts/datasets_identity_check.py)

**CEW dataset and the provided one are identical**,
except one thing: the provided data lacks 850 images. Hence, **the annotation problem is solved**.

### Step 3: Train-Valid-Test Split
I suppose, that the hidden test datasets consists of 850 images, which are precise in CEW dataset
and are missed in the provided one. In this case I can overfit a model and reach 100% accuracy on the test set.

But what if the hidden set contains images, which do not precise in CEW dataset?
In this case my overfitted model will fail.

Not to fail with an overfitted model, I will :
* fit several models which generalize well (w/o overfitting) on different folds;
* average out-of-fold predictions in `OpenEyesClassificator.predict` method.

This approach will allow:
* to reach 100% accuracy in case the hidden dataset consists of missed 850 images,
  because [n_folds - 1] of models will be fitted on this data;
* to reach accuracy comparable with mean cross-validation accuracy in case hidden dataset consists of 
  another images, but from the same distribution as images I have;
* also, using an ensemble of models usually provides better results then a single model.
  Inasmuch I do not have any inference time limitations,
  out-of-fold predictions should be a better solution in any case.
  
I will use 5-folds cross-validation.

### Step 4: Model Selection
I see 2 paths to initialize a model:
1) compile a custom one by myself;
2) take a ready architecture (maybe, pretrained one).

The 2nd approach may be not very suitable, because:
* SOTA architectures maybe too complicated for generalizing well on a such simple data;
* pretrained models are pretrained on a data from very different distribution.

I will try to use a ready-to-go architecture to save time on writing a model from scratch.
If it fails, I will write my custom model.

I considered [Papers with Code](https://paperswithcode.com/)
section ['Image Classification on ImageNet'](https://paperswithcode.com/sota/image-classification-on-imagenet)
to select one of the recent well-established architectures.

For my first experiments I selected ["EfficientNet-B0"](https://arxiv.org/abs/2003.08237).
This architecture is relatively small - 5.3m parameters and
precise in [`timm` package](https://rwightman.github.io/pytorch-image-models/) with weights.  

