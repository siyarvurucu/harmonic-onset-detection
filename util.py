import os
import numpy as np

def evalResults(noteids,nonnoteids,onsetSamples,sampleIdDict,result,orig,offset=12,twindow=24,H=128,verbose=True):
    """
    This function is for manually annotated data.  Compatible with annotation tools in the repository
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
    This function is for loading manually annotated data.
    
    filedir: annotation directory
    
    returns:
    sampleIdDict: dict of sample values (onset locations) to annotated onset IDs
    onsetSamples: list of onset locations
    IdOnsetDict: location of the onset inside the cropped annotated short sound clip
    note, nonnoteids: list of IDs of onsets, classified as 'note' and 'not note'
   
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


