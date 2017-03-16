import numpy as np
import pandas
from nilearn import plotting
import matplotlib.pyplot as plt
from nilearn.input_data import NiftiMasker
from sklearn.cross_validation import cross_val_score
from sklearn.svm import SVC
from sklearn.cross_validation import LeaveOneLabelOut
from nilearn.plotting import plot_stat_map, show
from sklearn.cross_validation import KFold
from sklearn.cross_validation import permutation_test_score
from sklearn.lda import LDA
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif

subjs = 'MES_022817_0'

datadir = '/Users/jamalw/Desktop/PNI/music_event_structures/'
fmri_filename = datadir + 'subjects/' + subjs + '/trans_filtered_func_data1.nii'
mask_filename = datadir + 'a1plus_2mm.nii' 
bg_img = datadir + 'MNI152_T1_2mm_brain.nii'

masker = NiftiMasker(mask_img=mask_filename, standardize=True)

# We give the masker a filename and retrieve a 2D array ready
# for machine learning with scikit-learn
fmri_masked = masker.fit_transform(fmri_filename)


# Create labels
labels = pandas.read_csv(datadir + 'subjects/' + subjs + '/s1model1.csv',header = 0)
target = []
for i in range(len(labels)):
    target.extend(np.repeat(labels.Genre[i],labels.Duration[i]))

target = np.asarray(target)

condition_mask = np.logical_or(target == 'Classical', target == 'Jazz')

# We apply this mask in the sampe direction to restrict the
# classification to the classical vs jazz discrimination
fmri_masked = fmri_masked[condition_mask]
target = target[condition_mask]

#select learning algorithms
svc = SVC(kernel='linear')
cv = KFold(n=len(fmri_masked),n_folds=5)
#feature_selection = SelectKBest(f_classif, k=500)

for train,test in cv:
    svc.fit(fmri_masked[train], target[train])
    prediction = svc.predict(fmri_masked[test])
    print((prediction == target[test]).sum() / float(len(target[test])))

# perform cross-validation
avg_cv_score = []
cv_score = cross_val_score(svc, fmri_masked, target, cv=cv)
avg_cv_scoreTemp = np.mean(cv_score)
avg_cv_score = np.append(avg_cv_score,avg_cv_scoreTemp)

# store model weights
resultsDir = datadir + 'subjects/' + subjs + '/model_weights_10fold_svm/'
coef_ = svc.coef_
coef_img = masker.inverse_transform(coef_)
#coef_img.to_filename(resultsDir + 'genre_svc_weights' + '.nii.gz')
#np.save(resultsDir + 'class_acc_' + subjs,avg_cv_score)

from sklearn.dummy import DummyClassifier
null_cv_scores = cross_val_score(DummyClassifier(), fmri_masked, target, cv=cv)
#null_cv_scores = permutation_test_score(svc, fmri_masked, target, cv=cv)
#np.save(resultsDir + 'null_cv_scores.npy', null_cv_scores[1])
