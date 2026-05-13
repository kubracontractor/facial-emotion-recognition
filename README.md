# facial-emotion-recognition
# Facial Emotion Recognition

A two-part AI project built for my final year at UCLan, classifying 
six facial emotions (Angry, Fear, Happy, Neutral, Sad, Surprise) 
using both traditional machine learning and deep learning.

The interesting outcome - traditional ML actually beat deep learning 
in this project, which taught me more about data size and model 
choice than any lecture did.

---

## Part 1 - Traditional ML (HOG + LBP with SVM and Random Forest)

I used two feature extraction methods (HOG and LBP) with two 
classifiers (SVM and Random Forest) across two datasets (JAFFE and CK+).

HOG worked significantly better than LBP because it captures 
edge direction and gradient information from facial muscle movements, 
whereas LBP only captures local texture and missed things like 
raised eyebrows or smile curves entirely.

**Best result: JAFFE + HOG + RF → 66% accuracy, F1: 0.69**

| Model | Accuracy | F1 |
|---|---|---|
| JAFFE + HOG + RF | 66% | 0.69 |
| JAFFE + HOG + SVM | 47% | 0.47 |
| CK+ + HOG + RF | 64% | 0.51 |
| CK+ + HOG + SVM | 63% | 0.53 |
| JAFFE + LBP + RF | 20% | 0.16 |
| JAFFE + LBP + SVM | 18% | 0.14 |

**Datasets:** JAFFE (213 images, controlled) and CK+ 
(484 labelled images, more diverse)

**Stack:** Python, scikit-learn, OpenCV, NumPy, Matplotlib, 
Google Colab

---

## Part 2 - Deep Learning (Custom CNN + VGG16 Transfer Learning)

Built a custom CNN from scratch and implemented transfer learning 
using VGG16 to address the limitations found in Part 1.

The custom CNN completely failed on CK+ - 341 training images is 
simply too small to train a CNN from scratch. It collapsed to 
predicting Neutral for everything. VGG16 with pre-trained ImageNet 
weights gave the best deep learning result at 40.56% on CK+, 
with meaningful recognition of Happy (F1 0.64) and Surprise (F1 0.44).

Neither model beat the 64% HOG + RF baseline from Part 1, which 
was the most honest finding of the whole project.

**Best deep learning result: VGG16 on CK+ → 40.56%, F1: 0.26**

| Model | Dataset | Accuracy | F1 |
|---|---|---|---|
| VGG16 Transfer Learning | CK+ | 40.56% | 0.26 |
| VGG16 Transfer Learning | FER-2013 | 24.35% | 0.22 |
| Custom CNN | FER-2013 | 23.52% | 0.21 |
| Custom CNN | CK+ | 25.17% | 0.07 |
| HOG + RF (Part 1 baseline) | CK+ | 64% | 0.61 |

**Datasets:** CK+ (341 training images) and FER-2013 
(~6,000 training images across 6 classes)

**Stack:** Python, TensorFlow, Keras, OpenCV, NumPy, 
Matplotlib, Google Colab

---

## Notebooks
- Part 1: https://colab.research.google.com/drive/1gZ8WIwDNlrocEQhHHdF_BvRKnuP3g-jh
- Part 2: https://colab.research.google.com/drive/1_uhAsqgFwW7puUbvuFwASu2ds2prBcGH
