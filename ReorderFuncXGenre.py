import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys
from nilearn.input_data import NiftiMasker
import soundfile as sf
import pickle

subj = sys.argv[1]
roi  = sys.argv[2]
roifilename = sys.argv[3]

datadir = '/jukebox/norman/jamalw/MES/'

# set data filenames
mask_filename  = datadir + 'data/' + roifilename
fmri1_filename = datadir + 'subjects/' + subj + '/analysis/run1.feat/trans_filtered_func_data.nii'
fmri2_filename = datadir + 'subjects/' + subj + '/analysis/run2.feat/trans_filtered_func_data.nii'

print("Masking data with " + roi + '...')
# mask data
masker = NiftiMasker(mask_img=mask_filename, standardize=True)
fmri1_masked = masker.fit_transform(fmri1_filename)
fmri1_masked = stats.zscore(fmri1_masked.T)
fmri2_masked = masker.fit_transform(fmri2_filename)
fmri2_masked = stats.zscore(fmri2_masked.T)

print("Loading song duration data...")
# load song data
songs1Dur = np.load(datadir + 'data/' + 'songs1Dur.npy')
songs2Dur = np.load(datadir + 'data/' + 'songs2Dur.npy')

print("Slicing functional data by song durations...")
# slice functional scan according to song durations
func_sliced1 = []
func_sliced2 = []
data1 = fmri1_masked
data2 = fmri2_masked
for i in range(len(songs1Dur)):
    func_sliced1.append([])
    func_sliced2.append([])
    func_sliced1[i].append(data1[:,0:songs1Dur[i]])
    func_sliced2[i].append(data2[:,0:songs2Dur[i]])
    data1 = data1[:,songs1Dur[i]:]
    data2 = data2[:,songs2Dur[i]:]

# create subject general song model for both experiments
exp1 = np.array([7, 12, 15,  2,  1,  0,  9,  3,  4,  5,  6, 13, 10,  8, 11, 14])
exp2 = np.array([3, 15,  4, 13,  2, 11,  0,  6,  7, 14,  1,  5, 10,  8, 12, 9])

print("Reordering functional data to match genre model...")
# reorder func data according to genre model
reorder1 = []
reorder2 = []
for i in range(len(exp1)):
    reorder1.append([])
    reorder2.append([])

for i in range(len(exp1)):
    reorder1[exp1[i]] = func_sliced1[i][0]
    reorder2[exp2[i]] = func_sliced2[i][0]

print("Saving reordered functional data to subject " + subj + " directory")
pickle.dump(reorder1, open(datadir + 'subjects/' + subj + '/data/' + 'reorder1' + roi + '.p','wb'))
pickle.dump(reorder2, open(datadir + 'subjects/' + subj + '/data/' + 'reorder2' + roi + '.p','wb'))
