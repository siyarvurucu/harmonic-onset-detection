import os
import numpy as np
def evalResults(noteids,nonnoteids,onsetSamples,sampleIdDict,result,orig,offset=12,twindow=24,H=128):
    """
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
                
                # ID = sampleIdDict[onsetLocation] # closest annotated onset id
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
                # onsetLocation = onsetSamples[np.argmin(np.abs(result[Pi]*H+offset*H-onsetSamples))]
                # ID = sampleIdDict[onsetLocation]
                print("closest onset to detected onset %d" % onsetLocation)
                print("ID: %s"%ID)
                if ID in matchedIDs:
                    print("this ID is already matched with another detection")
                print("detected onset at %d" % (result[Pi]*H+offset*H))
                print(result[Pi]*H+offset*H-onsetLocation)
                #ID = sampleIdDict[onsetLocation]
                
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
                    
                if ID not in noteids and ID not in nonnoteids:
                    print("this ID is not found")
                    print(ID)
                
            else:
                # onsetLocation = onsetSamples[np.argmin(np.abs(orig[Ni]*H+offset*H-onsetSamples))]
                # ID = sampleIdDict[onsetLocation]
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
            
    print("True positive:%d \n False Positive:%d \n False Negative:%d \n True Negative:%d \n Negative (na):%d \n Positive (na):%d \n Annotated Positive (missed):%d \n Annotated Negative (missed):%d" %(len(TP),len(FP),len(FN),len(TN),len(N),len(P),len(AP),len(AN)))
    #print("average score: %d" %np.mean(scores))
    # plt.vlines(scores[scoremask['tp']],ymin=0,ymax=1,color='c')
    # plt.vlines(scores[scoremask['fp']],ymin=0,ymax=1,color='r')
    # plt.vlines(scores[scoremask['fn']],ymin=0,ymax=1,color='y')
    # plt.vlines(scores[scoremask['tn']],ymin=0,ymax=1,color='m')
    # plt.vlines(scores[scoremask['p']],ymin=0,ymax=1,color='#FFBD33')
    # plt.vlines(scores[scoremask['n']],ymin=0,ymax=1,color='#924A03')
    # plt.show()
    return scoremask, TP, TN, FP, FN, P,N,AP,AN
    


def loadAnnotations(filedir):
    """
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


