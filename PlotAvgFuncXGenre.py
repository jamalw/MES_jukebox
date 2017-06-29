import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
import scipy as sp
from scipy import stats
import sys

roi = sys.argv[1]
subjs = ['MES_022817_0','MES_030217_0', 'MES_032117_1','MES_040217_0','MES_041117_0','MES_041217_0','MES_041317_0', 'MES_041417_0','MES_041517_0','MES_042017_0','MES_042317_0','MES_042717_0','MES_050317_0','MES_051317_0','MES_051917_0','MES_052017_0','MES_052017_1','MES_052317_0','MES_052517_0','MES_052617_0','MES_052817_0','MES_052817_1','MES_053117_0','MES_060117_0','MES_060117_1']
datadir = '/jukebox/norman/jamalw/MES/subjects/'

corrD3D = np.zeros((2511,2511,len(subjs)))
avgCorrD3D = np.zeros((16,16,len(subjs)))

for i in range(len(subjs)):
    corrD3D[:,:,i]   = np.load(datadir+str(subjs[i]) + '/data/corrD' + roi + '.npy')
    avgCorrD3D[:,:,i] = np.load(datadir+str(subjs[i]) + '/data/avgCorrD' + roi + '.npy')

meanCorrFull = np.mean(corrD3D,2)
meanCorrAvg  = np.mean(avgCorrD3D,2)

# compute average section of avgCorrD
corr_eye = np.identity(8)
classical_within  = meanCorrAvg[0:8,0:8]
classical_within_off  = classical_within[corr_eye == 0]
jazz_within       = meanCorrAvg[8:16,8:16]
jazz_within_off       = jazz_within[corr_eye == 0]
classJazz_between = meanCorrAvg[8:16,0:8]
classJazz_between_off = classJazz_between[corr_eye == 0]
jazzClass_between = meanCorrAvg[0:8,8:16]
jazzClass_between_off = jazzClass_between[corr_eye == 0]

labels = ["Classical","Jazz"]

plt.figure(1,facecolor="1")
plt.imshow(meanCorrFull,interpolation='none')
plt.colorbar()
plt.axis('tight')
ax = plt.gca()
plt.plot((ax.get_ylim()[0],ax.get_ylim()[1]),(ax.get_xlim()[1],ax.get_xlim()[0]),"k-")
pl.text(2800,2800,'N='+ str(len(subjs)),fontweight='bold')
plt.plot((-.5, 2511.5), (1255.5, 1255.5), 'k-')
plt.plot((1255.5, 1255.5), (-.5, 2511.5), 'k-')
pl.text(500,-70,labels[0])
pl.text(500,2800,labels[0])
pl.text(2550,500,labels[0],rotation='270')
pl.text(-300,500,labels[0],rotation='vertical')
pl.text(1800,-70,labels[1])
pl.text(-300,1800,labels[1],rotation='vertical')
pl.text(1800,2800,labels[1])
pl.text(2550,1800,labels[1],rotation='270')
pl.text(900.5, -200,'Full Correlation Matrix',fontweight='bold')
plt.savefig('/jukebox/norman/jamalw/MES/data/plots/FullCorrMat_N' + str(len(subjs)) + roi)

plt.figure(2,facecolor="1")
ax = plt.gca()
plt.imshow(meanCorrAvg,interpolation='none')
plt.plot((-.5, 15.5), (7.5, 7.5), 'k-')
plt.plot((7.5, 7.5), (-.5, 15.5), 'k-')
plt.colorbar()
plt.axis('tight')
plt.plot((ax.get_ylim()[0],ax.get_ylim()[1]),(ax.get_xlim()[1],ax.get_xlim()[0]),"k-")
plt.text(2.75,-1,labels[0],fontsize=15)
plt.text(2.75,17,labels[0],fontsize=15)
plt.text(15.65,2.75,labels[0],rotation='270',fontsize=15)
plt.text(-2,2.75,labels[0],rotation='vertical',fontsize=15)
plt.text(10.75,-1,labels[1],fontsize=15)
plt.text(-2,10.75,labels[1],rotation='vertical',fontsize=15)
plt.text(15.65,10.75,labels[1],rotation='270',fontsize=15)
plt.text(10.75,17,labels[1],fontsize=15)
plt.text(18,17,'N='+ str(len(subjs)),fontweight='bold',fontsize=15)
plt.text(19.5,8,'r',fontweight='bold',fontsize=15)
plt.text(1,-1.75,'Average Within-Song Correlation',fontweight='bold',fontsize=18)
plt.savefig('/jukebox/norman/jamalw/MES/data/plots/AvgCorrMat_N' + str(len(subjs)) + roi)

plt.figure(3,facecolor="1")
allComparisonsAvg = np.array([np.mean(classical_within_off),np.mean(jazz_within_off),np.mean(classJazz_between_off),np.mean(jazzClass_between_off)])
allComparisonsSem = np.array([stats.sem(classical_within_off),stats.sem(jazz_within_off),stats.sem(classJazz_between_off),stats.sem(jazzClass_between_off)])
N = 4
ind = np.arange(N)
width = 0.35
plt.bar(ind, allComparisonsAvg, width, color='k',yerr = allComparisonsSem,error_kw=dict(ecolor='lightseagreen',lw=3,capsize=0,capthick=0))
plt.ylabel('Pattern Similarity (r)',fontsize=15)
plt.title('Average Within and Between-Genre Similarity',fontweight='bold',fontsize=18)
labels = ['Classical vs Classical', 'Jazz vs Jazz', 'Jazz vs Classical', 'Classical vs Jazz']
plt.xticks(ind + width / 2,labels,fontsize=12)
plt.plot((-0.175,3.5),(0,0),'k-')
plt.savefig('/jukebox/norman/jamalw/MES/data/plots/AvgGenreSim_N' + str(len(subjs)) + roi) 

# Plot average Within song and Between song comparison
plt.figure(4,facecolor="1")
corr_eye = np.identity(16)
WithinBetwnSongAvgCorr = np.array([np.mean(meanCorrAvg[corr_eye == 1]),np.mean(meanCorrAvg[corr_eye == 0])])
WithinBetwnSongSemCorr = np.array([stats.sem(meanCorrAvg[corr_eye == 1]),stats.sem(meanCorrAvg[corr_eye == 0])])
N = 2
ind = np.arange(N)
width = 0.35
plt.bar(ind, WithinBetwnSongAvgCorr, width, color='k',yerr=WithinBetwnSongSemCorr,error_kw=dict(ecolor='lightseagreen',lw=3,capsize=0,capthick=0))
plt.ylabel('Pattern Similarity (r)',fontsize=15)
plt.title('Average Within- and Between-Song Similarity',fontweight='bold',fontsize=18)
labels = ['Same Song','Different Song']
plt.xticks(ind + width / 2,labels,fontsize=15)
plt.plot((-0.175,1.5),(0,0),'k-')
plt.savefig('/jukebox/norman/jamalw/MES/data/plots/AvgSongSim_N' + str(len(subjs)) + roi)

# Plot average Within song and Between song correlation for each genre
plt.figure(5,facecolor="1")
#compute average of within song/within genre correlations 
WithinSongCorr = meanCorrAvg[corr_eye == 1]
WithinSongAvgCorr = np.mean(meanCorrAvg[corr_eye == 1])
ClassicalWithinAvgOn = np.mean(WithinSongCorr[0:7])
JazzWithinAvgOn = np.mean(WithinSongCorr[8:15])
ClassicalWithinSemOn = stats.sem(WithinSongCorr[0:7])
JazzWithinSemOn = stats.sem(WithinSongCorr[8:15])

#compute average of between song/within genre correlations
corrEye8 = np.identity(8)
ClassicalBtwnAvgOff = np.mean(classical_within[corrEye8 == 0])
ClassicalBtwnSemOff = stats.sem(classical_within[corrEye8 == 0])
JazzBtwnAvgOff = np.mean(jazz_within[corrEye8 == 0])
JazzBtwnSemOff = stats.sem(jazz_within[corrEye8 == 0])

AvgAllGroups = np.array([ClassicalWithinAvgOn,ClassicalBtwnAvgOff,JazzWithinAvgOn,JazzBtwnAvgOff])
SemAllGroups = np.array([ClassicalWithinSemOn,ClassicalBtwnSemOff,JazzWithinSemOn,JazzBtwnSemOff])

N = 4
ind = np.arange(N)
width = 0.35
plt.bar(ind, AvgAllGroups, width, color='k',yerr=SemAllGroups,error_kw=dict(ecolor='lightseagreen',lw=3,capsize=0,capthick=0))
plt.ylabel('Pattern Similarity (r)',fontsize=15)
plt.title('Within- and Between-Song Similarity Within Genre',fontweight='bold',fontsize=18)
labels = ['Classical Within','Classical Between','Jazz Within', 'Jazz Between']
plt.xticks(ind + width / 2,labels,fontsize=14)
plt.plot((-0.175,1.5),(0,0),'k-')
pl.text(18,17,'N=2',fontweight='bold')
plt.savefig('/jukebox/norman/jamalw/MES/data/plots/WithinGenreOnOffDiag_N' + str(len(subjs)) + roi)



plt.show()


