import numpy as np

from scipy.signal import savgol_filter
import utilFunctions as UF

import json
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,5)

import IPython.display as ipd
from scipy.io.wavfile import read

from segmentation import getSegments
from transform import *

    
def MRSF(mX,B=[0,2000],a=51,b=3,size = 0.2):
    
    f = mX[1:,B[0]:B[1]]-mX[:-1,B[0]:B[1]]
    f[f<0] = 0
    sf = np.sum(f,axis=1)
    
    sfn = (sf)/np.max(np.abs(sf))
    
    sf_filtered = savgol_filter(sfn, a, b,deriv=0)
    #sf_filtered = signal.smooth(sfn,a)
    result=sf_filtered/np.max(sf_filtered)
    
    return np.asarray(result),sfn
    
def harmonicWeights(nH,c,p):
    """
    nH: number of harmonics considered for harmonic analysis. 10: f0,f1..f10
    c: strictness constant. 
    p: number of harmonics detected (out of nH)
    
    returns error multipliers for nH harmonics, ranging from 0 to 1
    """
    return np.exp(-np.arange(1,nH+1)/np.exp((nH+c-p)))

def candidSelection(sf,t=0.1,hw=50):
    """
    sf: spectral flux
    returns indexes of onset candidate frames
    """

    k=hw   # half window size for peak detection
    thres = t # relative energy threshold 
    sf = np.concatenate((np.mean(sf)*np.ones(k),sf,np.mean(sf)*np.ones(k)))
    det = np.zeros(sf.shape)
    
    means = [0 for i in range(k)]
    
    for i in range(sf.shape[0]-2*k):
        mean = (np.sum(sf[i:i+k])+np.sum(sf[i+k+1:i+2*k+1]))/(2*k)
        means.append(mean)
        if np.all(sf[i+k]>=sf[i:i+k])&np.all(sf[i+k]>=sf[i+k:i+2*k])&(sf[i+k]>(mean+thres)):
            det[i+k]=1
    means.extend([0 for i in range(k)])
    det = det[hw:-hw]
    
    det[:2*k] = 0
    det[-2*k:] = 0
    idx = np.nonzero(det)[0]   

    return idx
    
def ha(sf,sfn,mX,pX,params,verbose=[],onlySelected=False,hc=-2,div=8,L=30,fs=44100,gt=[]):
    """
    Applies harmonic analysis to onset candidates and returns the selected ones
    
    sf: spectral flux (filtered). used for candidate selection
    sfn: raw spectral flux. will be used for chord segmentation later
    mX, pX: freq. transform. mX is used, pX is for synthesis and observation purposes
    verbose: onset candids in this list will enable verbose mode.
    onlySelected: if True, only the candidates in verbose list will be processed.
    
    params: parameters used in freq. transform
    
    """
    
    M,N,H,B = params
    
    idx = candidSelection(sf,t=0.025,hw=25)    
    idx = np.concatenate((np.zeros(1),idx,np.array([sf.shape[0]])))
    idx_orig = idx.copy()
    mask = np.ones(idx.shape)
    mask[0]=0
    mask[-1]=0
    errors = np.zeros(mX.shape[0])
    scores = np.zeros(idx.shape)
    freqs = []
    
    tFlag = False
    vFlag = False # flag to enable prints and plots
    
    rms = np.sum(mX,axis=1)
    rms = rms-np.mean(rms)
    rms = rms/np.max(rms)
    rms = savgol_filter(rms,3,1)
    
    rms_t = -0.1
    
    # sending every onset candidate to harmonic analysis
    for i in range(len(idx)-2,0,-1):
                
        if onlySelected:
            if idx[i] not in verbose:
                continue
                
        b = int((idx[i]-(10240/H)) if (idx[i]>(idx[i-1]+(10240/H))) else idx[i-1])
        e = int((idx[i]+(10240/H)) if (idx[i]<(idx[i+1]-(10240/H))) else idx[i+1])
        
        
        if np.mean(rms[int(idx[i]):int(idx[i])+50])<rms_t:
            continue
        
        onst = int(idx[i]-b)
        pmX = np.copy(mX[b:e])
        

        if idx[i] in verbose:
            print("\nOnset candidate:")
            print("onset frame: %d" %idx[i])
            print("sf onset number: %d" %i)
            vFlag = True
            y = MRStftSynth(pmX,pX[b:e],M,H,B)
            print("synthesized sound")
            ipd.display(ipd.Audio(data=y, rate=fs))
            
        if vFlag:
            print("STFT around candidate")
            plt.pcolormesh(np.arange(pmX.shape[0]), np.arange(pmX.shape[1]), np.transpose(pmX))
            plt.show()
            
            print("filtered spectral flux")
            plt.plot(sf[b:e])
            plt.show()
            print("raw spectral flux")
            plt.plot(sfn[b:e])
            plt.show()
        
        allErrors,allf0s,pmXv = f0detection(pmX,pX[b:e],sfn[b:e],-100,10,onst,vFlag,hc,div,params,fs,tFlag)

        aL = np.min((e-idx[i]/2,L))  
        segments = getSegments(allf0s,allErrors,onst,pmX,vFlag)
        scores[i],freq,segmentScores = harmonicScore(segments,aL,vFlag,tFlag)
        freqs.append(freq)
        
        if scores[i]<1: # prevent rejected candidates from creating boundary for adjacent onset
            idx[i] = sf.shape[0]
            
        if vFlag:
            print("Score for this onset: %d" %scores[i])
            
        if tFlag and scores[i]<1:
            pred_time = np.abs(idx[i]*(H/fs))
            closest_gt_ind = np.argmin(pred_time-gt)[0]
            if np.abs(gt[closest_gt_ind]-pred_time)<0.05:
                if score[i]>1:
                    tp.append[idx[i]]
                if score[i]<1:
                    fn.append[idx[i]]
                    
                    print("STFT around onset")
                    plt.pcolormesh(np.arange(pmX.shape[0]), np.arange(pmX.shape[1]), np.transpose(pmX))
                    plt.show()
                    
                    y = MRStftSynth(pmXv,pX,M,H,B)
                    ipd.display(ipd.Audio(data=y, rate=fs))
                    
                    plt.pcolormesh(np.arange(pmXv.shape[0]), np.arange(pmXv.shape[1]), np.transpose(pmXv))
                    plt.show()

        vFlag = False
        tFlag = False
    
    avg = np.mean(scores)
    mask[scores<1] = 0
    result = idx_orig[mask==1]
    return idx_orig[1:-1],result,freqs,scores[1:-1]
    
def f0detection(pmX,pX,sfn,t,n,onset,verbose,hc,div,params,fs=44100,tFlag=False):
    
    M,N,H,B = params
    peaks = []
    allErrors = []
    allf0s = []
    pmXv = None
    binWidth = (fs/N)
     
    diff_idx0 = [(onset-8)//2 - 4,onset-8]
    diff_idx1 = [onset+4,(pmX.shape[0]+onset+4)//2]

    bOnsetEavg = np.mean(pmX[diff_idx0[0]:diff_idx0[1],:],axis=0)


    diff = np.mean(pmX[diff_idx1[0]:diff_idx1[1],:],axis=0)-bOnsetEavg
    diffr = np.repeat(diff[np.newaxis,:]>np.abs(bOnsetEavg/div),pmX.shape[0]-onset,axis=0)

    mask_t = 10
    right_m = 8
    for i in range(pmX[:onset].shape[0]):
        curPeaks = UF.peakDetection(pmX[i],t)
        peaks.append(curPeaks[curPeaks>(70/binWidth)])  
        
    for i in range(pmX[onset:].shape[0]):
        mask = np.ones(pmX.shape[1])*t
        curPeaks = UF.peakDetection(pmX[onset+i],t)
        curPeaks = curPeaks[curPeaks>(70/binWidth)]
        for c in curPeaks[curPeaks<500]:
            pMag = pmX[onset+i][c]
            pMag_tri = np.append(np.linspace(pMag,int(pMag-8*c/10),c//10)[::-1],np.linspace(pMag,int(pMag-8*c/10),right_m*c//10+1)[1:])-1
            #pMag = np.ones(mask[int(c-c/10):int(c+c/10)].size)*pMag
            mask[c-c//10:c+right_m*c//10] = np.amax((pMag_tri,mask[c-c//10:c+right_m*c//10]),axis=0)
        curPeaks_masked = curPeaks[pmX[onset+i][curPeaks]>mask[curPeaks]]
        curPeaks_masked_t = curPeaks_masked[diff[curPeaks_masked]>np.abs(bOnsetEavg/div)[curPeaks_masked]]
        peaks.append(curPeaks_masked_t)
        
    
    pmXv = np.copy(pmX)
    pmXv[onset:] = np.where(diffr,pmXv[onset:],np.min(pmXv[onset:]))
    
    if verbose: 
        print("remaining sound after elimination frequencies that do not contribute enough to energy increase for this candidate")
        
        y = MRStftSynth(pmXv,pX,M,H,B)
        ipd.display(ipd.Audio(data=y, rate=fs))
        
        plt.pcolormesh(np.arange(pmXv.shape[0]), np.arange(pmXv.shape[1]), np.transpose(pmXv))
        plt.vlines([diff_idx1[0],diff_idx1[1],diff_idx0[0],diff_idx0[1],onset],ymax=N/2,ymin=0) 
        plt.show()
    
    peaks = np.asarray(peaks)
 

     # evaluation of peaks
    
    for k in range(onset,pmX.shape[0],1):
    
        peaksinRange = peaks[k][np.logical_and(peaks[k]>(75/binWidth),peaks[k]<(2000/binWidth))]
        candidatesf0 = peaksinRange[np.argsort(pmX[k][peaksinRange])[::-1]][:5] 
        
                    
        if candidatesf0.size==0:
            allf0s.append([0])
            allErrors.append([1000])
            if tFlag:
                print("FN: no f0 candidate")
            continue

                
        possibleMissingf0 = []
        for p in peaks[k][np.logical_and(peaks[k]>(150/binWidth),peaks[k]<(2000/binWidth))][:5]:
            if np.min(np.abs(candidatesf0-p/2))>1:
                possibleMissingf0.append(p/2)
        candidatesf0 = np.append(candidatesf0,possibleMissingf0)

        errors = []
        nHarmonicsUsed = []

        for j in candidatesf0:
            cPeaks = []
            
                    
            nHarmonics = n
            nHarmonicsUsed.append(nHarmonics)

            # error calculation
            hErrors = []
            presentH = nHarmonics
            for i in np.arange(1,nHarmonics+1):
                cPeak = np.argmin(np.abs(peaks[k]-i*j))

                distance = peaks[k][cPeak]-i*j
                if np.abs(distance)>(i):
                    error = 30
                    presentH -= 1
                else:
                    error = np.abs(distance) 
                hErrors.append(error)
            
            
            hWeights = harmonicWeights(nHarmonics,hc,presentH)
            sum_error = np.sum(hWeights*hErrors)

            if verbose:
                if k == diff_idx1[1]:
                    print("candidate %d error: %d"%(j,sum_error))
            errors.append(sum_error)
    
        f0s = []
        f0sErrors = []
        
        # evaluating the errors for possible missing f0s
        for c in candidatesf0:
            possibleFreqs = np.abs(np.round(c/candidatesf0)-c/candidatesf0)<(1/candidatesf0)
            binsBelow = candidatesf0<c
            freqsBelow = candidatesf0[np.logical_and(binsBelow,possibleFreqs)]
            highIndex = np.where(candidatesf0==c)[0][0]

            freqFound = False

            for freq in np.sort(freqsBelow):

                if freqFound:
                    break

                lowIndex = np.where(candidatesf0==freq)[0][0]

                if candidatesf0[lowIndex] in f0s:
                    freqFound = True
                    break

                ratio = np.round(c/candidatesf0[lowIndex])
                if ratio==1:
                    break
                if ratio>3:
                    continue

                h = nHarmonicsUsed[lowIndex]          

                ratioIndex = np.where(ratio-np.arange(2,h+1)%ratio==ratio)[0][0]

                diffBeforeRatio = (ratio-np.arange(2,h+1)%ratio)[:ratioIndex]
                diffAfterRatio = np.minimum(ratio-np.arange(2,h+1)%ratio,np.arange(2,h+1)%ratio)[ratioIndex:]
                diff = np.append(diffBeforeRatio,diffAfterRatio)
                expectedError = np.sum(np.arange(1-(1/h),(1/h)-1e-4,-(1/h))*diff)

                lowError = errors[lowIndex]
                if lowError < (expectedError/2):
                    freqFound = True
                    f0s.append(candidatesf0[lowIndex])
                    f0sErrors.append(errors[lowIndex])

            if freqFound == False:
                f0s.append(c)
                f0sErrors.append(errors[highIndex])

        allErrors.append(f0sErrors)
        allf0s.append(f0s)
    return allErrors,allf0s,pmXv



def harmonicScore(segments,L,verbose,tFlag):
    if segments == None:
        return 0
    skipCountE = 0
    skipCountL = 0
    segmentScores = []
    freqs = {}
    for key in segments: # key: freqs
        freqs[key] = []
        for segment in segments[key]:  # each segment of this freq
            mags,startNfinish,errors = segment
            mags = np.asarray(mags)
            errors = np.asarray(errors)
            
            if (startNfinish[1]-startNfinish[0])<L:
                skipCountL += 1
                continue
            if np.mean(errors)>80:
                skipCountE += 1
                continue
            
            errors[errors<1] = 1
            segmentScore = np.sum((100+mags)/errors)/L
            freqs[key].append([segmentScore,startNfinish])
            segmentScores.append(segmentScore) 
            
    freqs = {k:freqs[k] for k in freqs.keys() if freqs[k]!=[]}
    score = sum(segmentScores)
    if tFlag:
        print("%d due to Length, %d due to Error, segment skipped" %(skipCountL,skipCountE))
        if len(segmentScores)==(skipCountL+skipCountE):
            print("no segments")
    if verbose:
        print("%d due to Length, %d due to Error, segment skipped" %(skipCountL,skipCountE))
        if len(segmentScores)==(skipCountL+skipCountE):
            print("no segments")
        
    return score,freqs,segmentScores
    
    
def od(filedir,hc=-1,div=7,L=20,a=51,res="low",gt=[],verbose=np.array([]),oS= False):
    fs, x = read(filedir)
    
    if res == "high":
        M = [8191,4095,2047,1023]
        N = 8192
        H = 128 # !! 128
        B = [[0,int(1000*N/fs)],[int(1000*N/fs),int(2000*N/fs)],[int(2000*N/fs),int(5000*N/fs)],[int(5000*N/fs),int(fs/2)]]
    
    if res == "med":
        M = [8191,2047,1023]
        N = 8192
        H = 128
        B = [[0,int(1000*N/fs)],[int(1000*N/fs),int(5000*N/fs)],[int(5000*N/fs),int(fs/2)]]
    
    if res == "low":
        M = [4095,2047]
        N = 4096
        H = 128
        B = [[0,int(2000*N/fs)],[int(2000*N/fs),int(fs/2)]]
    params= (M,N,H,B)
   
        
    mX,pX = fTransform(x,fs,params)
    
    sf,sfn= MRSF(mX,B=[0,int(2000*N/fs)],a=a,b=3)
     
     
    candids,result,freqs,scores = ha(sf,sfn,mX,pX,verbose=np.rint(verbose*fs/H).astype(int),onlySelected = oS,hc=hc,div=div,L=L,params=params,fs=fs,gt=gt)


    return result*H/fs,candids*H/fs,freqs