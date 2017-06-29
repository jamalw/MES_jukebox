import matplotlib.pyplot as plt
import pickle
import numpy as np
import sys
from scipy import stats

subj = sys.argv[1]
roi = sys.argv[2]

datadir = '/jukebox/norman/jamalw/MES/subjects/'

def corr2_coeff(A,B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:,None]
    B_mB = B - B.mean(1)[:,None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1);
    ssB = (B_mB**2).sum(1);

    # Finally get corr coeff
    return np.dot(A_mA,B_mB.T)/np.sqrt(np.dot(ssA[:,None],ssB[None]))

print("Loading subject " + subj + " reordered data...")
# take average and correlation for each subject pair
# load in subject data (16 lists of #VxTRs for each song)
subjData1 = pickle.load(open(datadir + subj + '/data/reorder1' + roi + '.p','rb'))
subjData2 = pickle.load(open(datadir + subj + '/data/reorder2' + roi + '.p','rb'))

# append average of each list to new array
avgsubjData1 = []
avgsubjData2 = []
for j in range(len(subjData1)):
    avgsubjData1.append(np.mean(subjData1[j],1))
    avgsubjData2.append(np.mean(subjData2[j],1))
# stack full and average data horizontally and vertically (respectively) to form #VxallTRs
subjData1Horz    = np.hstack(subjData1)
subjData2Horz    = np.hstack(subjData2)
avgsubjData1Horz = np.vstack(avgsubjData1)
avgsubjData2Horz = np.vstack(avgsubjData2)

print("Performing correlation...")
# perform correlation on full data and average data
corrD    = corr2_coeff(subjData1Horz.T,subjData2Horz.T)
avgCorrD = corr2_coeff(avgsubjData1Horz,avgsubjData2Horz)

print("Saving correlation data to subject " + subj + " directory")
np.save(datadir + str(subj) + '/data/' + 'corrD' + roi, corrD)
np.save(datadir + str(subj) + '/data/' + 'avgCorrD' + roi, avgCorrD)

# compute average section of avgCorrD and exclude diagonal to avoid incorporating same song info in average
corr_eye = np.identity(8)
classical_within  = avgCorrD[0:8,0:8]
classical_within  = classical_within[corr_eye == 0]
jazz_within       = avgCorrD[8:16,8:16]
jazz_within       = jazz_within[corr_eye == 0]
classJazz_between = avgCorrD[8:16,0:8]
classJazz_between = classJazz_between[corr_eye == 0]
jazzClass_between = avgCorrD[0:8,8:16]
jazzClass_between = jazzClass_between[corr_eye == 0]

plt.figure(1)
plt.imshow(corrD,interpolation='none')
plt.colorbar()
plt.axis('tight')

plt.figure(2)
plt.imshow(avgCorrD,interpolation='none')
plt.plot((-.5, 15.5), (7.5, 7.5), 'k-')
plt.plot((7.5, 7.5), (-.5, 15.5), 'k-')
plt.colorbar()
plt.axis('tight')

plt.figure(3)
allComparisonsAvg = np.array([np.mean(classical_within),np.mean(jazz_within),np.mean(classJazz_between),np.mean(jazzClass_between)])
allComparisonsSem = np.array([stats.sem(classical_within),stats.sem(jazz_within),stats.sem(classJazz_between),stats.sem(jazzClass_between)])
N = 4
ind = np.arange(N)
width = 0.35
labels = ['Classical vs Classical', 'Jazz vs Jazz', 'Jazz vs Classical', 'Classical vs Jazz']
plt.xticks(ind + width / 2, labels)
plt.bar(ind, allComparisonsAvg, width, color='k',yerr = allComparisonsSem,error_kw=dict(ecolor='lightseagreen',lw=3,capsize=0,capthick=0))
plt.plot((0,3.5),(0,0),'k-')

plt.show()

