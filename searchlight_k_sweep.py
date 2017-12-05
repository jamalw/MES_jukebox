import numpy as np
import brainiak.eventseg.event
from scipy.stats import norm,zscore,pearsonr,stats
from nilearn.image import load_img
import sys

#subjs = ['MES_022817_0','MES_030217_0','MES_032117_1','MES_040217_0','MES_041117_0','MES_041217_0','MES_041317_0','MES_041417_0','MES_041517_0','MES_042017_0','MES_042317_0','MES_042717_0','MES_050317_0','MES_051317_0','MES_051917_0','MES_052017_0','MES_052017_1','MES_052317_0','MES_052517_0','MES_052617_0','MES_052817_0','MES_052817_1','MES_053117_0','MES_060117_0','MES_060117_1']

subjs = ['MES_022817_0','MES_030217_0']
k_sweeper = [4,8,16,32,48,64]

others_idx  = sys.argv[1]
subj = subjs[int(others_idx)]
print('Subj: ', subj)

datadir = '/tigress/jamalw/MES/'
mask_img = load_img(datadir + 'data/mask_nonan.nii.gz')
mask = mask_img.get_data()
mask_reshape = np.reshape(mask,(91*109*91))
global_outputs_all = np.zeros((91,109,91))
results3d = np.zeros((91,109,91))

def searchlight(X1,X2,X3,X4,coords,K):
    
    """run searchlight 

       Create searchlight object and perform voxel function at each searchlight location
    
       Parameters
       ----------
       data1  : voxel by time ndarray (2D); leftout subject run 1
       data2  : voxel by time ndarray (2D); average of others run 1
       data3  : voxel by time ndarray (2D); leftout subject run 2
       data4  : voxel by time ndarray (2D); average of others run 2
       coords : voxel by xyz ndarray (2D, Vx3)
       K      : # of events for HMM (scalar)
       
       Returns
       -------
       3D data: brain (or ROI) filled with searchlight function scores (3D)

    """

    stride = 5
    radius = 5
    min_vox = 10
    nPerm = 1000
    SL_allvox = []
    SL_results = []
    for x in range(np.random.randint(stride),np.max(coords, axis=0)[0]+stride,stride):
        for y in range(np.random.randint(stride),np.max(coords, axis=0)[1]+stride,stride):
           for z in range(np.random.randint(stride),np.max(coords, axis=0)[2]+stride,stride):
              D = np.sqrt(np.square(coords - np.array([x,y,z])[np.newaxis,:]).sum(1))
              SL_vox = D <= radius
              if np.sum(SL_vox) < min_vox:
                 continue
              SL_within_across = HMM(X1[SL_vox,:],X2[SL_vox,:],X3[SL_vox,:],X4[SL_vox,:],K)
              if np.any(np.isnan(SL_within_across)):
                 continue
              SL_results.append(SL_within_across)
              SL_allvox.append(np.array(np.nonzero(SL_vox)[0]))
    voxmean = np.zeros((X1.shape[0], nPerm+1))
    vox_SLcount = np.zeros(X1.shape[0])
    for sl in range(len(SL_results)):
       voxmean[SL_allvox[sl],:] += SL_results[sl]
       vox_SLcount[SL_allvox[sl]] += 1
    voxmean = voxmean / vox_SLcount[:,np.newaxis]
    vox_z = (voxmean[:,0] - np.mean(voxmean[:,1:]))/np.std(voxmean[:,1:],axis=1) 
    return vox_z

def HMM(A,B,C,D,K):
    
    """fit hidden markov model
  
       Fit HMM to average data and cross-validate with leftout subject using within song and between song average correlations              

       Parameters
       ----------
       A: voxel by time ndarray (2D)
       B: voxel by time ndarray (2D)
       C: voxel by time ndarray (2D)
       D: voxel by time ndarray (2D)
       K: # of events for HMM (scalar)
 
       Returns
       -------
       z: z-score after performing permuted cross-validation analysis      

    """
    
    w = 20
    nPerm = 1000
    within_across1 = np.zeros(nPerm+1)
    within_across2 = np.zeros(nPerm+1)
    nTR = A.shape[1]

    # Fit to all but one subject
    ev1 = brainiak.eventseg.event.EventSegment(K)
    ev2 = brainiak.eventseg.event.EventSegment(K)
    ev1.fit(B.T)
    ev2.fit(D.T)
    events1 = np.argmax(ev1.segments_[0],axis=1)
    events2 = np.argmax(ev2.segments_[0],axis=1)   

    # Compute correlations separated by w in time
    corrs1 = np.zeros(nTR-w)
    corrs2 = np.zeros(nTR-w)
    for t in range(nTR-w):
        corrs1[t] = pearsonr(A[:,t],A[:,t+w])[0]
    _, event_lengths1 = np.unique(events1, return_counts=True)
       
    for t in range(nTR-w): 
        corrs2[t] = pearsonr(C[:,t],C[:,t+w])[0]
    _, event_lengths2 = np.unique(events2, return_counts=True)

    # Compute within vs across boundary correlations, for real and permuted bounds
    for p in range(nPerm+1):
        within1 = corrs1[events1[:-w] == events1[w:]].mean()
        within2 = corrs2[events2[:-w] == events2[w:]].mean()
        across1 = corrs1[events1[:-w] != events1[w:]].mean()
        across2 = corrs2[events2[:-w] != events2[w:]].mean()
        within_across1[p] = within1 - across1
        within_across2[p] = within2 - across2

        np.random.seed(p)
        perm_lengths1 = np.random.permutation(event_lengths1)
        np.random.seed(p)
        perm_lengths2 = np.random.permutation(event_lengths2)
        events1 = np.zeros(nTR, dtype=np.int)
        events2 = np.zeros(nTR, dtype=np.int)
        events1[np.cumsum(perm_lengths1[:-1])] = 1
        events2[np.cumsum(perm_lengths2[:-1])] = 1
        events1 = np.cumsum(events1)
        events2 = np.cumsum(events2)        

    within_across = (within_across1 + within_across2)/2 
    return within_across

for i in k_sweeper:
    #Load functional data and mask data
    np.random.seed(int(others_idx))
    print('Leftout:',subj)
    loo1 = load_img(datadir + 'subjects/' + subj + '/analysis/run1.feat/trans_filtered_func_data.nii').get_data()[:,:,:,0:2511]
    loo2 = load_img(datadir + 'subjects/' + subj + '/analysis/run2.feat/trans_filtered_func_data.nii').get_data()[:,:,:,0:2511]
    others1 = np.load(datadir + 'prototype/link/scripts/data/avg_others_run1/loo_' + str(others_idx) + '.npy')
    others2 = np.load(datadir + 'prototype/link/scripts/data/avg_others_run2/loo_' + str(others_idx) + '.npy')

    # create coords matrix
    x,y,z = np.mgrid[[slice(dm) for dm in loo1.shape[0:3]]]
    x = np.reshape(x,(x.shape[0]*x.shape[1]*x.shape[2]))
    y = np.reshape(y,(y.shape[0]*y.shape[1]*y.shape[2]))
    z = np.reshape(z,(z.shape[0]*z.shape[1]*z.shape[2]))
    coords = np.vstack((x,y,z)).T 
    coords_mask = coords[mask_reshape>0]
    print('Running Searchlight...')
    voxmean = searchlight(loo1[mask > 0,:], others1[mask > 0,:], loo2[mask > 0,:], others2[mask > 0,:], coords_mask,i) 
    results3d[mask>0] = voxmean
    print('Saving ' + subj + ' to Searchlight Folder')
    np.save(datadir + 'prototype/link/scripts/data/searchlight_output/HMM_searchlight_K_sweep/globals_loo_' + subj + '_K_' + str(i), results3d)



