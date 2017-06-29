# This script computes the pattern similarity within and between genres (classical and jazz) at every voxel in the brain via a Brainiak searchlight.

# Author: Jamal Williams
# Princeton Neuroscience Institute, Princeton University 2017

import numpy as np
from nilearn.image import load_img
import sys
from brainiak.searchlight.searchlight import Searchlight
from scipy import stats
import numpy as np
import nibabel as nib

# Take subject ID as input
subjs = ['MES_022817_0','MES_030217_0','MES_032117_1','MES_040217_0','MES_041117_0','MES_041217_0','MES_041317_0','MES_041417_0','MES_041517_0','MES_042017_0','MES_042317_0','MES_042717_0','MES_050317_0','MES_051317_0','MES_051917_0','MES_052017_0','MES_052017_1','MES_052317_0','MES_052517_0','MES_052617_0','MES_052817_0','MES_052817_1','MES_053117_0','MES_060117_0','MES_060117_1']

datadir = '/jukebox/norman/jamalw/MES/'
mask_img = load_img(datadir + 'data/MNI152_T1_2mm_brain_mask.nii')
mask_img = mask_img.get_data()
global_outputs_all = np.zeros((91,109,91,len(subjs)))

def corr2_coeff(AB,msk,myrad,bcast_var):
    if not np.all(msk):
        return None
    A,B = (AB[0], AB[1])
    A = A.reshape((-1,A.shape[-1]))
    B = B.reshape((-1,B.shape[-1]))
    corrAB = np.corrcoef(A.T,B.T)[16:,:16]
    corr_eye = np.identity(16)
    same_songs = corrAB[corr_eye == 1]
    diff_songs = corrAB[corr_eye == 0]	    
    avg_same_songs = np.mean(same_songs)
    avg_diff_songs = np.mean(diff_songs)
    same_song_minus_diff_song = avg_same_songs - avg_diff_songs
    # Compute difference score for permuted matrices
    np.random.seed(0)
    diff_perm_holder = np.zeros((100,1))
    for i in range(100):
        corrAB_perm = corrAB[np.random.permutation(16),:]
        same_songs_perm = corrAB_perm[corr_eye == 1]
        diff_songs_perm = corrAB_perm[corr_eye == 0]
        diff_perm_holder[i] = np.mean(same_songs_perm) - np.mean(diff_songs_perm)                
             
    z = (same_song_minus_diff_song - np.mean(diff_perm_holder))/np.std(diff_perm_holder)
    return z

for i in range(len(subjs)):
        counter = 0
        # Load functional data and mask data
        leftout1 = load_img(datadir + 'subjects/' + subjs[i] + '/data/avg_reorder1.nii')
        leftout2 = load_img(datadir + 'subjects/' + subjs[i] + '/data/avg_reorder2.nii')
        leftout1 = leftout1.get_data()
        leftout2 = leftout2.get_data()
        leftout = np.mean(np.array([leftout1,leftout2]),axis=0)
        
        others = np.zeros((902629,16,len(subjs)-1))
        print('Leftout:',i)
        for j in range(len(subjs)):
            if j != i: 
            # Calculate average of others
                 run1 = load_img(datadir + 'subjects/' + subjs[j] + '/data/avg_reorder1.nii')
                 run2 = load_img(datadir + 'subjects/' + subjs[j] + '/data/avg_reorder2.nii')
                 run1 = run1.get_data()
                 run2 = run2.get_data()
                 print('Subj:',j) 
            # Flatten data, then zscore data, then reshape data back into MNI coordinate space
                 avg_runs = np.mean(np.array([run1,run2]),axis=0)         
                 others[:,:,counter] = np.reshape(avg_runs,(91*109*91,16)) 
                 counter = counter + 1
            avg_others = np.reshape(np.mean(others,2),(91,109,91,16))
            
        np.seterr(divide='ignore',invalid='ignore')

        # Create and run searchlight
        sl = Searchlight(sl_rad=2,max_blk_edge=5)
        sl.distribute([leftout,avg_others],mask_img)
        sl.broadcast(None)
        print('Running Searchlight...')
        global_outputs = sl.run_searchlight(corr2_coeff)
        global_outputs_all[:,:,:,i] = global_outputs
        
# Plot and save searchlight results
global_outputs_avg = np.mean(global_outputs_all,3)
maxval = np.max(global_outputs_avg[np.not_equal(global_outputs_avg,None)])
minval = np.min(global_outputs_avg[np.not_equal(global_outputs_avg,None)])
global_outputs_avg = np.array(global_outputs_avg, dtype=np.float)
global_nonans = global_outputs_avg[np.not_equal(global_outputs_avg,None)]
global_nonans = np.reshape(global_nonans,(91,109,91))
min = np.min(global_nonans[~np.isnan(global_nonans)])
max = np.max(global_nonans[~np.isnan(global_nonans)])
img = nib.Nifti1Image(global_nonans, np.eye(4))
img.header['cal_min'] = min
img.header['cal_max'] = max
nib.save(img,'janice_results_n25_permuted.nii.gz')
np.save('janice_mat_n25_permuted',global_nonans)

print('Searchlight is Complete!')

import matplotlib.pyplot as plt
for (cnt, img) in enumerate(global_outputs_avg):
  plt.imshow(img,vmin=minval,vmax=maxval)
  plt.colorbar()
  plt.savefig(datadir + 'searchlight_images/' + 'img' + str(cnt) + '.png')
  plt.clf()


