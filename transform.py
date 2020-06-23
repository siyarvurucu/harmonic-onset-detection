import dftModel as DFT
from scipy.signal import get_window
import numpy as np

def dBsum(dBs,axis=None):
    totaldB = 20*np.log10(np.sum(10**(dBs/20),axis=axis))
    return totaldB
    
def MRStft(x,w,N,H,B,fs):  
    
    hM1 = []
    hM2 = []
    
    for i in range (len(B)):    
        hM1.append(int(np.floor((w[i].size+1)/2)))
        hM2.append(int(np.floor(w[i].size/2)))
        w[i] = w[i] / sum(w[i])                             
    
    x = np.append(np.zeros(max(hM2)),x)                
    x = np.append(x,np.zeros(max(hM1)))                 
    
    pin = max(hM1)   # hM1 is assumed to be bigger than H 
    pend = x.size- max(hM1)
    
    xmX = []
    xpX = []
    while pin<pend:   
        
        fmX = np.array([])
        fpX = np.array([])
    #-----analysis-----
        for i in range (len(B)):
            x1 = x[pin-hM1[i]:pin+hM2[i]] 
            mX, pX = DFT.dftAnal(x1, w[i], N)          
            fmX = np.append(fmX,mX[int(B[i][0]/(fs/N)):int(B[i][1]/(fs/N))])
            fpX = np.append(fpX,pX[int(B[i][0]/(fs/N)):int(B[i][1]/(fs/N))])
        fmX = np.append(fmX,mX[int(B[-1][1]/(fs/N))])
        fpX = np.append(fpX,pX[int(B[-1][1]/(fs/N))])
        
        xmX.append(fmX)
        xpX.append(fpX)
        pin += H 
    return np.asarray(xmX),np.asarray(xpX)

def MRStftSynth(mY,pY,M,H,B):  
     
    hM1 = []
    hM2 = []
    nFrames = mY[:,0].size
    
    for i in range (len(B)):    
        hM1.append(int(np.floor((M[i]+1)/2))) 
        hM2.append(int(np.floor(M[i]/2)))            
    
    y = np.zeros(nFrames*H + max(hM1) + max(hM2))   
     
    for i in range(len(B)):
        pin = max(hM1)   # hM1 is assumed to be bigger than H 
        pend = nFrames*H- max(hM1)
        yB = np.zeros(nFrames*H + max(hM1) + max(hM2))
    #-----synth-----
        for j in range(nFrames):
            y1 = DFT.dftSynth(mY[j,:], pY[j,:], M[i]) 
            yB[pin-hM1[i]:pin+hM2[i]] += H*y1    
            pin += H
        y += yB

    return y

def fTransform(x,fs,params=None):
    
    if params==None:
        M = [8192,4096,2048,1024]
        N = 16384
        H = 128
        B = [[0,int(1000*N/fs)],[int(1000*N/fs),int(2000*N/fs)],[int(2000*N/fs),int(5000*N/fs)],[int(5000*N/fs),int(fs/2)]]        
    else:    
        M,N,H,B = params
        
    window = 'blackman'
    fs = fs
   
    w = []
    for i in range(len(M)):
        w.append(get_window(window, M[i], fftbins=True))
        
    mX,pX = MRStft(x,w,N,H,B,fs)
    
    return mX,pX
    
