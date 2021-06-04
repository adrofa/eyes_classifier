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
except one thing: the provided data lacks 850 images.
I suppose, they are in the test (hidden) dataset.

