# MSCS 634 Lab 2: Classification Using KNN and RNN Algorithms

Rahul Solanki, MSCS-634-B01: Advanced Big Data and Data Mining, Professor Satish Penmatsa

## Purpose

Compare two distance-based classifiers, K-Nearest Neighbors (KNN) and Radius Neighbors (RNN), on the scikit-learn Wine dataset (178 wines, 13 chemical features, 3 cultivars), and see how the main parameter of each (k, radius) changes test accuracy.

| File | Contents |
|---|---|
| `MSCS_634_Lab_2.ipynb` | All code, outputs, plots, and the written comparison |
| `requirements.txt` | Python packages needed to run the notebook |

## Key insights

Stratified 80/20 split (142 training, 36 test wines, `random_state=42`), features unscaled:

| KNN k | 1 | 5 | 11 | 15 | 21 |
|---|---|---|---|---|---|
| Test accuracy | 0.778 | 0.806 | 0.806 | 0.806 | 0.806 |

| RNN radius | 350 | 400 | 450 | 500 | 550 | 600 |
|---|---|---|---|---|---|---|
| Test accuracy | 0.722 | 0.694 | 0.694 | 0.694 | 0.667 | 0.667 |

- **KNN beat RNN at every setting.** Best KNN 0.806 vs best RNN 0.722; the majority-class baseline is 0.389.
- **KNN is flat in k** (one test wine separates the best and worst k). Repeating the split 100 times shows k = 1 is best on average (0.759), so the ranking of k on a single 36-wine test set is mostly noise.
- **RNN gets worse as the radius grows.** At radius 600 the vote uses about 111 of the 142 training wines, and all 10 class_2 test wines are labeled class_1, the largest class.
- **One feature drives everything.** Proline supplies 99.7% of the squared Euclidean distance on raw data. class_0 has high proline and is almost always right; class_1 and class_2 overlap in proline and get confused.
- **Scaling fixes it.** With z-scored features KNN reaches 0.972 to 1.000, and RNN reaches 0.972 to 1.000 for radii of 2.5 to 3.5 standard units, but RNN needs its radius re-chosen in the new units (radius 2 leaves 16 test wines with no neighbors).
- **When to use which:** KNN is the safer default because every prediction uses exactly k neighbors and k does not depend on units. RNN fits data where distance has a real meaning and density is even, and it can flag points with no neighbors as outliers.

## Challenges and decisions

- **Unscaled features for the main runs:** the radius values in the assignment (350 to 600) only make sense in raw units, so Steps 2 and 3 use raw features. Scaling is tested separately at the end of Step 4.
- **Stratified split:** keeps the 33/40/27% class mix in both the training and test sets (12/14/10 test wines).
- **No-neighbor safeguard:** `RadiusNeighborsClassifier` raises an error when a test point has no training point inside the radius, so `outlier_label="most_frequent"` is set. It was never needed at radii 350 to 600 (every test wine had at least 4 neighbors), only in the scaled check.
- **Small test set:** 36 test wines means one wine = 2.8% accuracy, so the notebook repeats both sweeps over 100 random splits before drawing conclusions about k and radius.

## How to run

Python 3.11 with the packages in `requirements.txt` (`pip install -r requirements.txt`). Open `MSCS_634_Lab_2.ipynb` and run all cells; the dataset is loaded from scikit-learn, so no data files are needed.
