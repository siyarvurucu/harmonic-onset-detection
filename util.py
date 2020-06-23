import os
import numpy as np
import json
import mir_eval
from scipy.io.wavfile import read
import quickPlayer


def getCloseOnsets(ci,onsets,combine_window):
    closeOnsetsidx = np.where(np.logical_and(onsets>onsets[ci],onsets<(onsets[ci]+combine_window)))[0]
    return closeOnsetsidx

def getOnsets(filedir,combine_window = 0.025,dataset = "gs",fs=44100,mergeMethod = "first"):
    """
    combines the onsets closer than combine_window, by averaging them.
    dataset: "gs" GuitarSet or "mc" musiccritic
    mergeMethod: Pick "first", "last", or "avg" of very close onsets. This only matters for GuitarSet, as onsets of each string annotated separately
    fs should be provided for "mc", not all recordings are sampled at 44100
    """  
    
    if dataset == "mc":
        combine_window = combine_window*fs
    
    onsets = []
    for x in os.listdir(filedir):
        if x[:5]=='onset':
            with open(filedir +"/"+ x, 'r') as filehandle:
                values = json.load(filehandle)
                onsets.extend(values)
    onsets = np.sort(np.asarray(onsets))
    
    ci = 0
    mergedOnsets = []
    mergedGroups = []

    closeOnsets = np.array([onsets[ci]])
    while ci < (onsets.size):
        closeOnsetsidx = getCloseOnsets(ci,onsets,combine_window)
        if closeOnsetsidx.size > 0:
            ci = closeOnsetsidx[-1]
            closeOnsets = np.append(closeOnsets,onsets[closeOnsetsidx])
        else:
            if mergeMethod == "first":
                mergedOnsets.append(closeOnsets[0])
            if mergeMethod == "avg":
                mergedOnsets.append(np.mean(closeOnsets))
            if mergeMethod == "last":
                mergedOnsets.append(closeOnsets[-1])
                
            mergedGroups.append(closeOnsets)
            ci += 1
            if ci<(onsets.size):
                closeOnsets = np.array([onsets[ci]])
    mergedOnsets = np.asarray(mergedOnsets)   
    
    if dataset == "mc":
        mergedOnsets = mergedOnsets/fs
    
    return mergedOnsets,mergedGroups

def summarizeScores(scores,chordfiles): 
                
    fscores_solo = []
    precision_solo = []
    recall_solo = []
    fscores_chords =[]
    precision_chords =[]
    recall_chords = []
    for i in range(len(scores)):
        if scores[i][0] in chordfiles:
            fscores_chords.append(scores[i][1]['F-measure'])
            precision_chords.append(scores[i][1]['Precision'])
            recall_chords.append(scores[i][1]['Recall'])
        else:
            
            fscores_solo.append(scores[i][1]['F-measure'])
            precision_solo.append(scores[i][1]['Precision'])
            recall_solo.append(scores[i][1]['Recall'])
    
    fscores_solo = np.asarray(fscores_solo)
    precision_solo = np.asarray(precision_solo)
    recall_solo = np.asarray(recall_solo)
    fscores_chords =np.asarray(fscores_chords)
    precision_chords =np.asarray(precision_chords)
    recall_chords = np.asarray(recall_chords)

    print("Solo files: %d"%fscores_solo.size)
    print("F-score: %f"%np.mean(fscores_solo))
    print("Precision: %f"%np.mean(precision_solo))
    print("Recall: %f"%np.mean(recall_solo))

    print("Chords files: %d"%fscores_chords.size)
    print("F-score: %f"%np.mean(fscores_chords))
    print("Precision: %f"%np.mean(precision_chords))
    print("Recall: %f"%np.mean(recall_chords))

    print("Overall: %d"%len(scores))
    print("F-score: %f"%((np.sum(fscores_solo)+np.sum(fscores_chords))/len(scores)))
    print("Precision: %f"%((np.sum(precision_solo)+np.sum(precision_chords))/len(scores)))
    print("Recall: %f"%((np.sum(recall_solo)+np.sum(recall_chords))/len(scores)))

# to change the offset or windowsize without re-running
def recalcScore(scores,offset=0,window=0.05):
    newScores = []
    for score in scores:
        filename = score[0]
        gt = score[2]
        result= score[3]
        result = np.array(result)
        nscore = mir_eval.onset.evaluate(gt,result+offset,window = window)
        newScores.append([filename,nscore,gt,result])
    return newScores
    
def evalResults(noteids,nonnoteids,onsetSamples,sampleIdDict,result,orig,offset=12,twindow=24,H=128,verbose=True):
    """
    This function is for files annotated with spectral flux assistance. 
    twindow: truth window (number of samples)
    offset: constant value added to detections (result) before evaluating
    
    """

    TP = []
    FP = []
    FN = []
    TN = []
    N = []
    P = []
    AP = []
    AN = []
    scoremask = {'tp': [],'fp': [],'tn': [],'fn': [],'n': [],'p': []}
    annotatedmask = np.ones(onsetSamples.shape)
    
    matchedIDs = [] 
    
    for i in range(orig.size):
        
        if np.min(np.abs(orig[i]-result))==0:
            
            Pi = np.where(orig[i]-result == 0)[0][0] 
            
            idx = np.argmin(np.abs(result[Pi]*H+offset*H-onsetSamples))
            onsetLocation = onsetSamples[idx]
            
            ID = sampleIdDict[onsetLocation]
            
            if (np.abs((result[Pi]*H)+offset*H-onsetLocation)<(twindow)) and (ID not in matchedIDs):
                
                matchedIDs.append(ID)
                
                annotatedmask[idx]=0
                
                if ID in noteids:
                    TP.append([ID,result[Pi]+offset])
                    scoremask['tp'].append(i)
                if ID in nonnoteids:
                    FP.append([ID,result[Pi]+offset])
                    scoremask['fp'].append(i)
                if ID not in noteids and ID not in nonnoteids:
                    print("this ID is not found")
                    print(ID)
            else:
                if verbose:
                    print("closest onset to detected onset %d" % onsetLocation)
                    print("ID: %s"%ID)
                    if ID in matchedIDs:
                        print("this ID is already matched with another detection")
                    print("detected onset at %d" % (result[Pi]*H+offset*H))
                    print(result[Pi]*H+offset*H-onsetLocation)
                
                P.append(["na",result[Pi]+offset])
                scoremask['p'].append(i)
        else:
            Ni = i
            
            idx = np.argmin(np.abs(orig[Ni]*H+offset*H-onsetSamples))
            onsetLocation = onsetSamples[idx]
            ID = sampleIdDict[onsetLocation]
            
            if (np.min(np.abs((orig[Ni]*H)+offset*H-onsetLocation))<(twindow)) and (ID not in matchedIDs):

                matchedIDs.append(ID)
                annotatedmask[idx]=0
                
                if ID in noteids:
                    FN.append([ID,orig[Ni]+offset])
                    scoremask['fn'].append(i)
                if ID in nonnoteids:
                    TN.append([ID,orig[Ni]+offset])
                    scoremask['tn'].append(i)
                    
                if verbose:     
                    if ID not in noteids and ID not in nonnoteids:
                        print("this ID is not found")
                        print(ID)
                
            else:
                if verbose:
                    print("closest onset to detected onset %d" % onsetLocation)
                    print("ID: %s"%ID)
                    if ID in matchedIDs:
                        print("this ID is already matched with another detection")
                    print("detected onset at %d" % (orig[Ni]*H+offset*H))
                    print(orig[Ni]*H+offset*H-onsetLocation)
                    
                N.append(["na",orig[Ni]+offset])
                scoremask['n'].append(i)
    
    for onsetLocation in onsetSamples[annotatedmask==1]:
        ID = sampleIdDict[onsetLocation]
        if ID in noteids:
            AP.append([ID,onsetLocation/H])
        if ID in nonnoteids:
            AN.append([ID,onsetLocation/H])
    if verbose:    
        print("True positive:%d \n False Positive:%d \n False Negative:%d \n True Negative:%d \n Negative (na):%d \n Positive (na):%d \n Annotated Positive (missed):%d \n Annotated Negative (missed):%d" %(len(TP),len(FP),len(FN),len(TN),len(N),len(P),len(AP),len(AN)))
    return scoremask, TP, TN, FP, FN, P,N,AP,AN
    


def loadAnnotations(filedir):
    """
    This function is for files annotated with spectral flux assistance.
    
    filedir: annotation directory
    
    returns:
    sampleIdDict: dict of sample values (onset locations) to annotated onset IDs
    onsetSamples: list of onset locations
    IdOnsetDict: location of the onset inside the cropped annotated short sound clip
    note, nonnoteids: list of IDs of onsets, classified as 'note' and 'not a note'
   
    """
    
    f = open(filedir + "/onsets.txt", "r")
    data = f.read() 
    f.close()
    
    sampleIdDict = {}
    IdOnsetDict = {}
    onsetSamples = []
    
    for entry in data.split("\n")[:-1]:
        ID,onset,sample = entry.split(" ")
        onsetSamples.append(int(sample))
        sampleIdDict[int(sample)] = ID
        IdOnsetDict[ID]=onset
        
    onsetSamples = np.asarray(onsetSamples)
    
    annotations = {}
    for folder in os.listdir(filedir):
        if folder =='onsets.txt':
            continue
            
        annotations[folder] = os.listdir(filedir + "/"+ folder)

        nonnoteids = []
        noteids = []

        for key in list(annotations.keys()):
            if key == 'note':
                noteids = annotations[key]
            else:
                nonnoteids.extend(annotations[key])
    
    return sampleIdDict, IdOnsetDict, onsetSamples, noteids, nonnoteids 

def separateFiles(dir,dataset="gs"):

    with open(dir+"list_files.txt","r") as f:
        filelist = json.load(f)
        
    chordfiles = []
    solofiles = []  
    
    if dataset == "gs":  
        for i in range(360): # separating solo and accompaniment recording names
            filename = filelist[i]
            
            if filename == "Annotations":
                continue
            
            if filename.split("_")[2] == "comp":
                chordfiles.append(filename)
            else:
                solofiles.append(filename)
                
    if dataset == "mc":
     
        with open(dir+"chords_list","r") as f:
            chordfiles = f.read().split("\n")
        chordfiles = [c for c in chordfiles if c!=""]
        
        for file in filelist:
            if file not in chordfiles:
                solofiles.append(file)
    return solofiles,chordfiles
    
def play(score,dataset="mc",withCandids=False):
    """
    dataset: "mc" or "gs"
    withCandids: Also displays all the onset candidates
    """
    if dataset=="mc":
        filedir = "musiccritic_cropped/"+score[0]+".wav"
        
    if dataset=="gs":
        filedir = "guitarset/"+score[0]+".wav"
        
    fs,x = read(filedir)
    x = x/np.max(np.abs(x))
    
    gt = score[2]
    gt = [[" ",g*fs] for g in gt]
    
    pred = score[3]
    pred = [[int(p*fs/128),p*fs] for p in pred]
    
    if withCandids == True:
        candid = score[5]
        candid = [[i,candid[i]*fs] for i in range(len(candid))]
        quickPlayer.quickPlayer(filedir,[x],[gt,pred,candid],["gt","pred","sf"],1,False)
    else:
        quickPlayer.quickPlayer(filedir,[x],[gt,pred],["gt","pred"],1,False)

    
def details(score,window=0.05):
    
    offset = 0.01
    candids = candid = score[5]
    gt = candid = score[2]
    pred = candid = score[3]
    
    tp = []
    tn = []
    fp = []
    fn = []

    fni = []
    for i in range(len(candids)):
        closest = np.argsort(np.abs(candids[i]-gt))[0]
        
        if candids[i] in pred:
            
            if np.abs(candids[i]+offset-gt[closest])<(0.05):
                tp.append(i)
            else:
                fp.append(i)
        else:
            
            if np.abs(candids[i]+offset-gt[closest])<(0.05):
                fn.append(i)
            else:
                tn.append(i) 
    return tp,tn,fp,fn 