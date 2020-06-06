import numpy as np
from scipy.signal import get_window
from scipy.signal import find_peaks_cwt
from scipy.signal import savgol_filter
import utilFunctions as UF
import dftModel as DFT
import json
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,5)
from matplotlib import collections  as mc
import IPython.display as ipd

def MRStft(x,w,N,H,B,fs):  
    
    hM1 = []
    hM2 = []
    
    for i in range (len(B)):    
        hM1.append(int(np.floor((w[i].size+1)/2)))        # half analysis window size by rounding
        hM2.append(int(np.floor(w[i].size/2)))            # half analysis window size by floor
#         pin_array.append(max(H, hM1[i]))                  # init sound pointer in middle of anal window       
#         pend_array.append(x.size - max(H, hM1[i]))        # last sample to start a frame
        w[i] = w[i] / sum(w[i])                             # normalize analysis windows
    
    x = np.append(np.zeros(max(hM2)),x)                 # add zeros at beginning to center first window at sample 0
    x = np.append(x,np.zeros(max(hM1)))                 # add zeros at the end to analyze last sample
    
    pin = max(hM1)   # bigger than H is assumed
    pend = x.size- max(hM1)
    
    xmX = []
    xpX = []
    #xfX = []
    while pin<pend:   
        
        fmX = np.array([])
        #fX = np.array([])
        fpX = np.array([])
    #-----analysis-----
        for i in range (len(B)):
            x1 = x[pin-hM1[i]:pin+hM2[i]] 
#             fx1 = scipy.fft.fft(x1*w[i],N)[:N//2+1]
#             afx1 = abs(fx1)/x1.size
#             afx1[afx1<np.finfo(float).eps] = np.finfo(float).eps 
            #mX = 20*np.log10(afx1)
            mX, pX = DFT.dftAnal(x1, w[i], N)                # compute dft
            fmX = np.append(fmX,mX[int(B[i][0]/(fs/N)):int(B[i][1]/(fs/N))])
            fpX = np.append(fpX,pX[int(B[i][0]/(fs/N)):int(B[i][1]/(fs/N))])
#             fX = np.append(fX,fx1[int(B[i][0]/(fs/N)):int(B[i][1]/(fs/N))])
        xmX.append(fmX)
#         xfX.append(fX)
        xpX.append(fpX)
        pin += H 
    return np.asarray(xmX),np.asarray(xpX)

def MRStftSynth(mY,pY,M,H,B,fs):  # pY > fY
    
    hM1 = []
    hM2 = []
    nFrames = mY[:,0].size
    
    for i in range (len(B)):    
        hM1.append(int(np.floor((M[i]+1)/2)))        # half analysis window size by rounding
        hM2.append(int(np.floor(M[i]/2)))            # half analysis window size by floor
#         pin_array.append(max(H, hM1[i]))                  # init sound pointer in middle of anal window       
#         pend_array.append(x.size - max(H, hM1[i]))        # last sample to start a frame
    
    y = np.zeros(nFrames*H + max(hM1) + max(hM2))                 # add zeros at the end to analyze last sample
     
    for i in range(len(B)):
        pin = max(hM1)   # bigger than H is assumed
        pend = nFrames*H- max(hM1)
        yB = np.zeros(nFrames*H + max(hM1) + max(hM2))
    #-----synth-----
        for j in range(nFrames):
#             y1 = scipy.fft.ifft(np.concatenate((fY[j,:],fY[j,:][::-1])),M[i]).real
            y1 = DFT.dftSynth(mY[j,:], pY[j,:], M[i])       # compute idft
            yB[pin-hM1[i]:pin+hM2[i]] += H*y1                   # overlap-add to generate output sound
            pin += H
        y += yB

    return y

def fTransform(x):
    window = 'blackman'
    fs = 44100
    M = [8191,4096,2048,1024]
    N = 16384
    H = 128
    w = []
    B = [[0,400],[400,800],[800,2000],[2000,22053]]
    for i in range(len(M)):
        w.append(get_window(window, M[i], fftbins=True))
        
    mX,pX = MRStft(x,w,N,H,B,fs)
    
    return mX,pX
    
def MRSF(mX,pX,B=[0,8193],a=51,b=3):
    
    f = mX[1:,B[0]:B[1]]-mX[:-1,B[0]:B[1]]
    f[f<0] = 0
    sf = np.sum(f,axis=1)
    sf[:20] = np.mean(sf)
    sf[-20:] = np.mean(sf)
    
    sfn = sf/np.max(sf)
    #plt.plot(sf)
    #plt.show()
    sf_filtered = savgol_filter(sf, a, b,deriv=0)
    #plt.plot(sf_filtered)
    #plt.show()
    result=sf_filtered/np.max(sf_filtered) # ??
    #result[:20] =
    #plt.plot(result)
    #plt.show()
    return np.asarray(result),sfn,mX,pX
    
def harmonicWeights(nH,c,p):
    return np.exp(-np.arange(1,nH+1)/np.exp((nH+c-p)))   # error multplier for i th harmonic for detected harmonics


def candidSelection(sf):

    det = np.zeros(sf.shape)
    k=20
    means = [0 for i in range(k)]
    
    for i in range(sf.shape[0]-2*k):
        mean = (np.sum(sf[i:i+k])+np.sum(sf[i+k+1:i+2*k+1]))/(2*k)
        means.append(mean)
        if np.all(sf[i+k]>=sf[i:i+k])&np.all(sf[i+k]>=sf[i+k:i+2*k])&(sf[i+k]>(mean+0.02)):  # most basic
            det[i+k]=1
    means.extend([0 for i in range(k)])
    idx = np.nonzero(det)[0]   

    return idx
    
def odf(sf,sfn,mX,pX,params=None,verbose=[],onlySelected=False,hc=0,div=4,L=5):
    #(threshold=0.5, smooth=0.0, pre_avg=0.0, post_avg=0.0, pre_max=0.0, post_max=0.0, combine=0.03, delay=0.0, online=False, fps=100, **kwargs)
    #result = OnsetPeakPickingProcessor(threshold=0.3,pre_avg=0.005, post_avg=0.005, pre_max=0.005, post_max=0.005,fps=344.53125)(func)
    
    fs = 44100
    
    idx = candidSelection(sf)
    
    #idx = idx + (4096/(2*128)) # fixfrequency 𝑓0) for the whole file. We then segment the pitch contour into stable regions. The stable regions most likely correspond to notes of the melody. ing index for sample format
    
    idx = np.concatenate((np.zeros(1),idx,np.array([sf.shape[0]])))
    mask = np.ones(idx.shape)
    mask[0]=0
    mask[-1]=0
    errors = np.zeros(mX.shape[0])
    scores = np.zeros(idx.shape)
    
    
    
    vFlag = False # flag to enable prints and plots
    
    for i in range(1,len(idx)-1):
        
        if onlySelected:
            if idx[i] not in verbose:
                continue
        
        b = int((idx[i]-(9600/128)) if (idx[i]>(idx[i-1]+(9600/128))) else idx[i-1])
        e = int((idx[i]+(9600/128)) if (idx[i]<(idx[i+1]-(9600/128))) else idx[i+1])
#         try:
        onst = int(idx[i]-b)
        pmX = np.copy(mX[b:e])
        
        if idx[i] in verbose:
            print("ONSET")
            print("onset frame: %d" %idx[i])
            print("sf onset number: %d" %i)
            vFlag = True
            M = [8191,4096,2048,1024]
            N = 16384
            H = 128
            B = [[0,400],[400,800],[800,2000],[2000,22053]]
            y = MRStftSynth(pmX,pX[b:e],M,H,B,44100)
            ipd.display(ipd.Audio(data=y, rate=44100))
            
        if vFlag:
            plt.pcolormesh(np.arange(pmX.shape[0]), np.arange(pmX.shape[1]), np.transpose(pmX))
            plt.show()
        
        allErrors,allf0s,rmX = f0detection(pmX,pX[b:e],sfn[b:e],-90,10,onst,vFlag,16384,hc,div)
        
        
        
        if vFlag:
            xdata = []
            ydata = []
            srea = []
            for k in range(len(allf0s)):  # k frame, j freqs
                for j in range(len(allf0s[k])):
                    xdata.append(idx[i]+k)
                    ydata.append(allf0s[k][j])
                    srea.append(mX[b:e][k][int(allf0s[k][j])])
                    
            #fig = plt.figure(figsize=(20,5))
            plt.scatter(xdata, ydata, (2**10)*10**(np.asarray(srea)/20), alpha=0.5)
            plt.show()
        
       
        segments = newgetSegments(allf0s,allErrors,onst,mX[b:e],vFlag)
        scores[i] = harmonicScore(segments,L,vFlag)
        
        if vFlag:
            print("Score for this onset: %d" %scores[i])
        
        # old eval 

        minerrors = []
        for err in allErrors:
            minerrors.append(min(err))          
        errors[int(idx[i]):e] = minerrors[onst:int(e-b)]
       
        vFlag = False
#         if (sum(minerrors[onst+16:])/len(minerrors[onst+16:]))>20:
#                 #print(idx[i],np.std(f0s[onset:]))
#                 mask[i]=0
              
        # debug stuff
#          if idx[i]==4070:
#             print(onst)
#             print(b)
#             print(e)
#             plt.plot(minerrors)
#             print((sum(minerrors[onst+16:])/len(minerrors[onst+16:])))
#             plt.show()
#             print(pmX.shape)
#             plt.pcolormesh(np.arange(mX[b:e].shape[0]), np.arange(4097), np.transpose(mX[b:e]))
#             plt.show()
#             plt.pcolormesh(np.arange(rmX.shape[0]), np.arange(4097), np.transpose(rmX))
#             plt.show()

    #meanScore= np.mean(scores)
    #print(meanScore)
    
    avg = np.mean(scores)
    mask[scores<1] = 0
    result = idx[mask==1]
    result = result   # WindowSize/2 /H
    return idx[1:-1],result,errors,scores[1:-1]#,means
    
    
def dBsum(dBs,axis=None):
    totaldB = 20*np.log10(np.sum(10**(dBs/20),axis=axis))
    return totaldB
    
    
def f0detection(pmX,pX,sfn,t,n,onset,verbose,N,hc,div):
    
    M = [8191,4096,2048,1024]
    N = 16384
    H = 128
    B = [[0,400],[400,800],[800,2000],[2000,22053]]
    fs = 44100
    peaks = []
    allErrors = []
    allf0s = []
    
    if verbose:
        fa = np.zeros(pmX.shape[0]-1)
        for i in range(len(B)):
            f = pmX[1:,int(B[i][0]*(N/fs)):int(B[i][1]*(N/fs))]-pmX[:-1,int(B[i][0]*(N/fs)):int(B[i][1]*(N/fs))]
            ivf = f.copy()
            ivf[ivf>0] = 0
            f[f<0] = 0
            #sfB.append(np.sum(f,axis=1))
            print(B[i][0],B[i][1])
            f = np.sum(f,axis=1)
            # ivf = np.sum(ivf,axis=1)
            plt.plot(f)
            plt.show()
            fa = fa+f
           
        # print("sf all")
        # plt.plot(fa)
        # plt.show() 
        
        # print("filtered")
        # sf_filtered = savgol_filter(fa, 51, 3,deriv=0)
        # sf_filtered=sf_filtered/np.max(sf_filtered)
        # plt.plot(sf_filtered)
        # plt.show()
        
        fa -= f
        print("sf lower")
        plt.plot(fa)
        plt.show() 
        print("filtered")
        sf_filtered = savgol_filter(fa, 51, 3,deriv=0)
        sf_filtered=sf_filtered/np.max(sf_filtered)
        plt.plot(sf_filtered)
        plt.show()
    # srms = []
    
    # for i in range(len(B)):
        # srmsb = dBsum(pmX[:,int(B[i][0]*(N/fs)):int(B[i][1]*(N/fs))],axis=1)
        # srms.append(srmsb)
        # if verbose:
            # print(B[i][0],B[i][1])
            # plt.plot(srmsb)
            # plt.vlines([onset],ymin=min(srmsb),ymax=max(srmsb))
            # plt.show()
            
    # srmsall =  dBsum(pmX,axis=1)    
    # srmsdiff = srmsall[1:]-srmsall[:-1]
     
    # if verbose:
        # plt.plot(srmsall)
        # plt.vlines([onset],ymin=min(srmsall),ymax=max(srmsall))
        # plt.show()
        # plt.plot(srmsdiff)
        # plt.vlines([onset],ymin=min(srmsdiff),ymax=max(srmsdiff))
        # plt.show()
        
#     diff_idx0 = np.argmin(srmsdiff[:onset]<0)[0][-1]
    #diff_idx0 = np.where(srmsdiff[:onset]<0)[0][-1]
    #diff_idx1 = onset+np.where(srmsdiff[onset:]<0)[0][0]
    #diff_idx1 = onset+np.argmax(srmsall[onset:])
    
    # if (onset-diff_idx0)<8:   # 32 16
        # if verbose:
            # print("diff_idx0 was %d" %diff_idx0)
        # diff_idx0 = onset-8
        

    # if (diff_idx1-onset)<16:  
        # if verbose:
            # print("diff_idx1 was %d" %diff_idx1)
        # diff_idx1 = onset + 16

    # if (diff_idx1-onset)>50:
        # print("diff1 50")
    # if (-diff_idx0+onset)>50:
        # print("diff0 50")
    

    # minpeakind = find_peaks_cwt(-savgol_filter(sfn,11,3), np.arange(1,11))
    # maxpeakind = find_peaks_cwt(savgol_filter(sfn,11,3), np.arange(1,11))
    # try:
        # diff_idx0 = minpeakind[minpeakind<onset][np.argmin(np.abs(minpeakind[minpeakind<onset]-onset))]
    # except:
        # diff_idx0 = onset-8
    # try:
        # diff_idx1 = minpeakind[minpeakind>onset][np.argmin(np.abs(minpeakind[minpeakind>onset]-onset))]
    # except:
        # diff_idx1 = onset+16
    
    # if verbose:
        # plt.plot(sfn)
        # plt.vlines([diff_idx1,diff_idx0],ymin=min(sfn),ymax=max(sfn))
    diff_idx0 = onset-4
    diff_idx1 = onset+4
    
    diff = np.max(pmX[diff_idx1:,:],axis=0)-pmX[diff_idx0,:]#pmX[diff_idx0]    np.mean(pmX[onset-48:onset-16]    
    diffr = np.repeat(diff[np.newaxis,:]>np.abs(pmX[diff_idx0]/div),pmX.shape[0]-onset,axis=0)
    # np.abs(pmX[diff_idx0]/div)
#     pmX[onset:] = np.where(diffr,pmX[onset:],np.min(pmX[onset:]))
    
    
    # peak detection
    for i in range(pmX[:onset].shape[0]): # 
        #t = mX[i][np.argsort(mX[i])[::-1]][100]
        peaks.append(UF.peakDetection(pmX[i],t))
        
    for i in range(pmX[onset:].shape[0]):
        #t = mX[i][np.argsort(mX[i])[::-1]][100]
        upeaks = UF.peakDetection(pmX[onset+i],t)
        upeaks = upeaks[diff[upeaks]>np.abs(pmX[diff_idx0]/div)[upeaks]] # np.abs(pmX[diff_idx0]/div)[upeaks]
        peaks.append(upeaks)
    
    if verbose:
        pmX[onset:] = np.where(diffr,pmX[onset:],np.min(pmX[onset:]))
        y = MRStftSynth(pmX,pX,M,H,B,44100)
        ipd.display(ipd.Audio(data=y, rate=44100))
        
        plt.pcolormesh(np.arange(pmX.shape[0]), np.arange(pmX.shape[1]), np.transpose(pmX))
        plt.vlines([diff_idx1,diff_idx0,onset],ymax=N/2,ymin=0) # diff_idx0
        plt.show()
    
    peaks = np.asarray(peaks)
    
    binWidth = (44100/N)
    
    for k in range(pmX.shape[0]):
        
    #     if len(peaks[k][np.logical_and(peaks[k]>14,peaks[k]<200)])<5:
    #         # possible silence
    #         topf0.append(0)
    #         topErrors.append([1000])
    #         continue

        # why first 5? use magnitude?
        peaksinRange = peaks[k][np.logical_and(peaks[k]>(75/binWidth),peaks[k]<(2000/binWidth))]
        candidatesf0 = peaksinRange[np.argsort(pmX[k][peaksinRange])[::-1]][:10]  # 5 highest mag. peaks in range
        if verbose:
                if k == diff_idx1:
                    plt.plot(pmX[k][:2000])
                    plt.show()
                    
        if candidatesf0.size==0:
            allf0s.append([0])
            allErrors.append([1000])
            if verbose:
                print("No f0 candidate at frame %d" %k)
            continue

                
        possibleMissingf0 = []
        for p in peaks[k][np.logical_and(peaks[k]>(150/binWidth),peaks[k]<(2000/binWidth))][:5]:
            if np.min(np.abs(candidatesf0-p/2))>1:
                possibleMissingf0.append(p/2)
        candidatesf0 = np.append(candidatesf0,possibleMissingf0)

        errors = []
        nHarmonicsUsed = []
    #     errorsMag = []
    #     loudness = 2**(mX[0][candidatesf0.astype(int)]-(min(mX[0][candidatesf0.astype(int)])-1))/6
    #     loudnessM = np.sum(loudness)/loudness

        for j in candidatesf0:

    #         if j<8 or j>100:    # those bins depend on fft size 
    #             errors.append(1000)
    #             errorsMag.append(1000)
    #             nHarmonicsUsed.append(0)
    #             continue

            cPeaks = []
            remaining=peaks[k].copy()
            if verbose:
                if k == diff_idx1:
                    print("%d.th frame" %k)
                    print("f0 cand. %d" %j)
                    print("peaks")
                    print(remaining)
                    
            # determining number of harmonics for error calc
#             nPossibleH = int(peaks[k][-1]/j)+1
#             nHarmonics = min(n,nPossibleH)
            nHarmonics = n
            nHarmonicsUsed.append(nHarmonics)

            # error calculation
            hErrors = []
            presentH = nHarmonics
            for i in np.arange(1,nHarmonics+1):
                cPeak = np.argmin(np.abs(remaining-i*j))

                #hWeight = (nHarmonics-(i-1))/(nHarmonics)
                distance = peaks[k][cPeak]-i*j
                #inharm = i*j*(1+(36*520e-6))**0.5 - i*j
                     # ((3*i)+inharm) or distance<(-3*i):
                if np.abs(distance)>(3*i):
                    error = 30
                    presentH -= 1
                else:
                    error = np.abs(distance) 
                hErrors.append(error)#*hWeight
            
            
            hWeights = harmonicWeights(nHarmonics,hc,presentH)
            sum_error = np.sum(hWeights*hErrors)
                #print(error)
            if verbose:
                if k == diff_idx1:
                    print("error: %d"%sum_error)
            errors.append(sum_error)
            cindex = np.where(candidatesf0==j)[0][0]
    
    
        f0s = []
        f0sErrors = []
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

                if h==0:
                    print("h = 0")
                    print(lowIndex)
                    print(highIndex)
                    print("cand")
                    print(candidatesf0[lowIndex])
                    print(candidatesf0[highIndex])
                    print(candidatesf0)
                    break            

        #             try:
                ratioIndex = np.where(ratio-np.arange(2,h+1)%ratio==ratio)[0][0]
        #             except:
        #                 print("ratio problem")
        #                 print(ratio)

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

    return allErrors,allf0s,pmX



def newgetSegments(allf0s,allErrors,onset,mX,verbose):
    lines = []
    segments_open = {}
    segments_closed = {}
    for i in range(len(allf0s[onset:])):
        open_increasing = {key:False for key in segments_open}
        
        for j in range(len(allf0s[onset+i])):
    
            if allErrors[onset+i][j]<100:
            
                if allf0s[onset+i][j] in segments_open:
                    #segments_open[allf0s[onset+i][j]][1] += 1
                    harmonicsMag = dBsum(mX[onset+i][(allf0s[onset+i][j]*np.arange(1,10)).astype(int)])
                    segments_open[allf0s[onset+i][j]][0].append(harmonicsMag)
                    segments_open[allf0s[onset+i][j]][2].append(allErrors[onset+i][j])
                    open_increasing[allf0s[onset+i][j]] = True
                    
                elif (allf0s[onset+i][j]+1) in segments_open:
                    #segments_open[allf0s[onset+i][j]+1][1] += 1
                    
                    segments_open[allf0s[onset+i][j]] = segments_open[allf0s[onset+i][j]+1]
                    del segments_open[allf0s[onset+i][j]+1]
                    del open_increasing[allf0s[onset+i][j]+1]
                    
                    harmonicsMag = dBsum(mX[onset+i][((allf0s[onset+i][j])*np.arange(1,10)).astype(int)])
                    segments_open[allf0s[onset+i][j]][0].append(harmonicsMag)
                    segments_open[allf0s[onset+i][j]][2].append(allErrors[onset+i][j])
                    open_increasing[allf0s[onset+i][j]] = True   
                    
                elif (allf0s[onset+i][j]-1) in segments_open:
                    segments_open[allf0s[onset+i][j]] = segments_open[allf0s[onset+i][j]-1]
                    del segments_open[allf0s[onset+i][j]-1]
                    del open_increasing[allf0s[onset+i][j]-1]
                    
                    harmonicsMag = dBsum(mX[onset+i][((allf0s[onset+i][j])*np.arange(1,10)).astype(int)])
                    segments_open[allf0s[onset+i][j]][0].append(harmonicsMag)
                    segments_open[allf0s[onset+i][j]][2].append(allErrors[onset+i][j])
                    open_increasing[allf0s[onset+i][j]] = True   
                    
                else: 
                    harmonicsMag = dBsum(mX[onset+i][(allf0s[onset+i][j]*np.arange(1,10)).astype(int)])
                    segments_open[allf0s[onset+i][j]] = [[harmonicsMag],[i],[allErrors[onset+i][j]]]
                    open_increasing[allf0s[onset+i][j]] = True
                    
        for key in open_increasing:
            if open_increasing[key]==False:
                segments_open[key][1].append(i)
                finishedSegment = segments_open.pop(key)
                #segments_open[key][0].append(mX[int(onset+i-1)][int(key)])
                if key not in segments_closed: #
                    segments_closed[key] = []
                segments_closed[key].append(finishedSegment)
                
                
                if verbose:
                    line = [(finishedSegment[1][0],key),(finishedSegment[1][1],key)]
                    lines.append(line)
        
    for key in open_increasing:
            if open_increasing[key]==True:
                segments_open[key][1].append(i)
                finishedSegment = segments_open.pop(key)
                #segments_open[key][0].append(mX[int(onset+i-1)][int(key)])
                if key not in segments_closed:
                    segments_closed[key] = []
                segments_closed[key].append(finishedSegment)
                
                if verbose:
                    line = [(finishedSegment[1][0],key),(finishedSegment[1][1],key)]                   
                    lines.append(line)
        
                
    if verbose:
        print("Total n of segments: %d" %len(lines))
        lc = mc.LineCollection(lines, linewidths=2)
        fig, ax = plt.subplots()
        ax.add_collection(lc)
        ax.autoscale()
        ax.margins(0.1)
        plt.show()
        
    if 0 in segments_closed:
        print("gotcha")
    return segments_closed
                    
def getSegments(allf0s,allErrors,onset,mX,verbose):
    """
    returns starting magnitude and length of segments in a dict
    onset: relative onset
    b: beginning 
    allf0s, allErrors: comes from harmonic presence analysis
    """
    lines = []
    segments_open = {}
    segments_closed = {}
    for i in range(len(allf0s[onset:])):
        open_increasing = {key:False for key in segments_open}
        
        for j in range(len(allf0s[onset+i])):
            
            if allErrors[onset+i][j]<100:
                
                if allf0s[onset+i][j] in segments_open:
                    #segments_open[allf0s[onset+i][j]][1] += 1
                    harmonicsMag = dBsum(mX[onset+i][(allf0s[onset+i][j]*np.arange(1,10)).astype(int)])
                    segments_open[allf0s[onset+i][j]][0].append(harmonicsMag)
                    segments_open[allf0s[onset+i][j]][2].append(allErrors[onset+i][j])
                    open_increasing[allf0s[onset+i][j]] = True
                    
                elif (allf0s[onset+i][j]+1) in segments_open:
                    #segments_open[allf0s[onset+i][j]+1][1] += 1
                    harmonicsMag = dBsum(mX[onset+i][((allf0s[onset+i][j]+1)*np.arange(1,10)).astype(int)])
                    segments_open[allf0s[onset+i][j]+1][0].append(harmonicsMag)
                    segments_open[allf0s[onset+i][j]+1][2].append(allErrors[onset+i][j])
                    open_increasing[allf0s[onset+i][j]+1] = True
                    
                elif (allf0s[onset+i][j]-1) in segments_open:
                    #segments_open[allf0s[onset+i][j]-1][1] += 1
                    harmonicsMag = dBsum(mX[onset+i][((allf0s[onset+i][j]-1)*np.arange(1,10)).astype(int)])
                    segments_open[allf0s[onset+i][j]-1][0].append(harmonicsMag)
                    segments_open[allf0s[onset+i][j]-1][2].append(allErrors[onset+i][j])
                    open_increasing[allf0s[onset+i][j]-1] = True
                    
                else: 
                    harmonicsMag = dBsum(mX[onset+i][(allf0s[onset+i][j]*np.arange(1,10)).astype(int)])
                    segments_open[allf0s[onset+i][j]] = [[harmonicsMag],[i],[allErrors[onset+i][j]]]
                    open_increasing[allf0s[onset+i][j]] = True
                    
        for key in open_increasing:
            if open_increasing[key]==False:
                segments_open[key][1].append(i)
                finishedSegment = segments_open.pop(key)
                #segments_open[key][0].append(mX[int(onset+i-1)][int(key)])
                if key not in segments_closed: #
                    segments_closed[key] = []
                segments_closed[key].append(finishedSegment)
                
                
                if verbose:
                    line = [(finishedSegment[1][0],key),(finishedSegment[1][1],key)]
                    lines.append(line)
        
    for key in open_increasing:
            if open_increasing[key]==True:
                segments_open[key][1].append(i)
                finishedSegment = segments_open.pop(key)
                #segments_open[key][0].append(mX[int(onset+i-1)][int(key)])
                if key not in segments_closed:
                    segments_closed[key] = []
                segments_closed[key].append(finishedSegment)
                
                if verbose:
                    line = [(finishedSegment[1][0],key),(finishedSegment[1][1],key)]                   
                    lines.append(line)
        
                
    if verbose:
        print("Total n of segments: %d" %len(lines))
        lc = mc.LineCollection(lines, linewidths=2)
        fig, ax = plt.subplots()
        ax.add_collection(lc)
        ax.autoscale()
        ax.margins(0.1)
        plt.show()
        
    if 0 in segments_closed:
        print("gotcha")
    return segments_closed

def harmonicScore(segments,L,verbose):
    skipCountE = 0
    skipCountL = 0
    segmentScores = []
    for key in segments: # key: freqs
        for segment in segments[key]:  # each segment of this freq
            mags,startNfinish,errors = segment
            mags = np.asarray(mags)
            errors = np.asarray(errors)
            
            if (startNfinish[1]-startNfinish[0])<L:
                skipCountL += 1
                continue
            if np.mean(errors)>40:
                skipCountE += 1
                continue
            
            errors[errors==0] = 1
            #mags = 10**(mags/20)
            mags = 2**((mags+80)/6)
            segmentScore = mags*(1/errors)

            #magb = magb+90
            segmentScores.append(np.sum(segmentScore))
            
    score = sum(segmentScores)#/len(segmentScores) # is mean good?
    if verbose:
        print("%d due to Length, %d due to Error, segment skipped" %(skipCountL,skipCountE))
        if len(segmentScores)==(skipCountL+skipCountE):
            print("no segments")
    return score
    
    
def od(filedir):
    offset = 16*128/44100
    # folder,filename = filedir.split("/")
    fs, x = UF.wavread(filedir)
    if fs != 44100:
        print(" %s sampling rate: %d" %(filedir,fs))
    #xeq = ess.EqualLoudness()(x)
    mX,pX = fTransform(x)
    # with open(folder+"/stfts"+filename+".json", 'w') as f:
        # json.dump([mX.tolist(),pX.tolist()], f) 
    sf,sfn,mX,pX = MRSF(mX,pX,B=[0,850],a=51,b=3)
    orig,result,errors,scores = odf(sf,sfn,mX,pX,verbose=[],onlySelected = False,hc=-3,div=10,L=30)
    return (result*(128/fs))+offset