# This script computes the HMM fit to the actual song durations at every voxel in the brain using Brainiaks searchlight function and event segmentation function.

# Author: Jamal Williams
# Princeton Neuroscience Institute, Princeton University 2017

print('importing packages')
import numpy as np
from nilearn.image import load_img
import sys
from brainiak.searchlight.searchlight import Searchlight
from scipy.stats import norm, zscore,pearsonr,stats
import numpy as np
import nibabel as nib
import brainiak.eventseg.event
from scipy.signal import gaussian,convolve
from sklearn import decomposition

# Take subject ID as input
#subjs = ['MES_022817_0','MES_030217_0','MES_032117_1','MES_040217_0','MES_041117_0','MES_041217_0','MES_041317_0','MES_041417_0','MES_041517_0','MES_042017_0','MES_042317_0','MES_042717_0','MES_050317_0','MES_051317_0','MES_051917_0','MES_052017_0','MES_052017_1','MES_052317_0','MES_052517_0','MES_052617_0','MES_052817_0','MES_052817_1','MES_053117_0','MES_060117_0','MES_060117_1']

subjs = ['MES_022817_0','MES_030217_0','MES_032117_1']

datadir = '/jukebox/norman/jamalw/MES/'
mask_img = load_img(datadir + 'data/MNI152_T1_2mm_brain_mask.nii')
mask_img = mask_img.get_data()
global_outputs_all = np.zeros((91,109,91,len(subjs)))

def fit_HMM(AB,msk,myrad,bcast_var):
    if not np.all(msk):
        return None
    w = 5
    nPerm = 1000
    within_across = np.zeros(nPerm+1)
    A,B = (AB[0], AB[1])
    nTR = A.shape[3]
    A = A.reshape(-1,1255)    

    # Fit to all but one subject
    ev = brainiak.eventseg.event.EventSegment(8)
    B = B.reshape(-1,1255)
    ev.fit(B.T)
    events = np.argmax(ev.segments_[0],axis=1)

    # Compute correlations separated by w in time
    corrs = np.zeros(nTR-w)
    for t in range(nTR-w):
        print('t: ',t)
        corrs[t] = np.nan_to_num(pearsonr(A[:,t],A[:,t+w])[0])
    _, event_lengths = np.unique(events, return_counts=True) 
 
    # Compute within vs across boundary correlations, for real and permuted bounds
    np.random.seed(0)
    for p in range(nPerm+1):
        within = corrs[events[:-w] == events[w:]].mean()
        across = corrs[events[:-w] != events[w:]].mean()
        within_across[p] = within - across

        perm_lengths = np.random.permutation(event_lengths)
        events = np.zeros(nTR, dtype=np.int)
        events[np.cumsum(perm_lengths[:-1])] = 1
        events = np.cumsum(events)
        
    z = (within_across[0] - np.mean(within_across[1:]))/np.std(within_across[1:])
    print(z)
    return z

for i in range(len(subjs)):
        # Load functional data and mask data
        print('Leftout:',i,subjs[i])
        leftout = load_img(datadir + 'subjects/' + subjs[i] + '/data/avg_reordered_both_runs.nii.gz').get_data()[:,:,:,0:1255]
        
        others = np.zeros((91,109,91,1255),dtype=float)
        #print('Leftout:',i,subjs[i])
        for j in range(len(subjs)):
            if j != i:
                # Calculate average of others
                 print('Subj:',j)
                 data = np.array(load_img(datadir + 'subjects/' + subjs[j] + '/data/avg_reordered_both_runs.nii.gz').get_data()[:,:,:,0:1255])
                 #print('Subj:',j)
                 others += data/len(subjs)-1

        np.seterr(divide='ignore',invalid='ignore')

        # Create and run searchlight
        print('Hello')
        sl = Searchlight(sl_rad=2,max_blk_edge=5)
        print('Hello 2')
        sl.distribute([leftout,others],mask_img)
        print('Hello 3')
        sl.broadcast(None)
        print('Running Searchlight...')
        global_outputs = sl.run_searchlight(fit_HMM)
        print('Adding ' + subjs[i] + ' to Global Outputs')
        global_outputs_all[:,:,:,i] = global_outputs

# Plot and save searchlight results
global_outputs_avg = np.mean(global_outputs_all,3)
maxval = np.max(global_outputs_avg[np.not_equal(global_outputs_avg,None)])
minval = np.min(global_outputs_avg[np.not_equal(global_outputs_avg,None)])
global_outputs_avg = np.array(global_outputs_avg, dtype=np.float)
global_nonans = global_outputs_avg[np.not_equal(global_outputs_avg,None)]
global_nonans = np.reshape(global_nonans,(91,109,91))
min1 = np.min(global_nonans[~np.isnan(global_nonans)])
max1 = np.max(global_nonans[~np.isnan(global_nonans)])
img = nib.Nifti1Image(global_nonans, np.eye(4))
img.header['cal_min'] = min1
img.header['cal_max'] = max1
nib.save(img,'HMM_searchlight_results_n25_permuted.nii.gz')
np.save('HMM_searchlight_mat_n25_permuted',global_nonans)

print('Searchlight is Complete!')        
