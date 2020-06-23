import matplotlib.pyplot as plt
from matplotlib import collections as mc
plt.rcParams["figure.figsize"] = (20,5)
from transform import dBsum
import numpy as np
def getSegments(allf0s,allErrors,onset,mX,verbose):
    """
    create segments for f0s that has harmonics with less than the error threshold.
    """    
    if len(allf0s) == 0:
        return None
    error_thres = 200
    lines = []
    segments_open = {}
    segments_closed = {}
    open_increasing = {}
    for i in range(len(allf0s)):
        open_increasing = {key:False for key in segments_open}
        
        for j in range(len(allf0s[i])):
    
            if allErrors[i][j]<error_thres:
                
                if allf0s[i][j] in segments_open:
                    harmonicsMag = dBsum(mX[onset+i][(allf0s[i][j]*np.arange(1,10)).astype(int)])
                    segments_open[allf0s[i][j]][0].append(harmonicsMag)
                    segments_open[allf0s[i][j]][2].append(allErrors[i][j])
                    open_increasing[allf0s[i][j]] = True
                    
                elif (allf0s[i][j]+1) in segments_open:
                    
                    segments_open[allf0s[i][j]] = segments_open[allf0s[i][j]+1]
                    del segments_open[allf0s[i][j]+1]
                    del open_increasing[allf0s[i][j]+1]
                    
                    harmonicsMag = dBsum(mX[onset+i][((allf0s[i][j])*np.arange(1,10)).astype(int)])
                    segments_open[allf0s[i][j]][0].append(harmonicsMag)
                    segments_open[allf0s[i][j]][2].append(allErrors[i][j])
                    open_increasing[allf0s[i][j]] = True   
                    
                elif (allf0s[i][j]-1) in segments_open:
                    segments_open[allf0s[i][j]] = segments_open[allf0s[i][j]-1]
                    del segments_open[allf0s[i][j]-1]
                    del open_increasing[allf0s[i][j]-1]
                    
                    harmonicsMag = dBsum(mX[onset+i][((allf0s[i][j])*np.arange(1,10)).astype(int)])
                    segments_open[allf0s[i][j]][0].append(harmonicsMag)
                    segments_open[allf0s[i][j]][2].append(allErrors[i][j])
                    open_increasing[allf0s[i][j]] = True   
                    
                else: 
                    harmonicsMag = dBsum(mX[onset+i][(allf0s[i][j]*np.arange(1,10)).astype(int)])
                    segments_open[allf0s[i][j]] = [[harmonicsMag],[i],[allErrors[i][j]]]
                    open_increasing[allf0s[i][j]] = True
                    
        for key in open_increasing:
            if open_increasing[key]==False:
                segments_open[key][1].append(i)
                finishedSegment = segments_open.pop(key)
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
                if key not in segments_closed:
                    segments_closed[key] = []
                segments_closed[key].append(finishedSegment)
                
                if verbose:
                    line = [(finishedSegment[1][0],key),(finishedSegment[1][1],key)]                   
                    lines.append(line)
        
                
    if verbose:
        print("Total number of segments: %d" %len(lines))
        lc = mc.LineCollection(lines, linewidths=2)
        fig, ax = plt.subplots()
        ax.add_collection(lc)
        ax.autoscale()
        ax.margins(0.1)
        plt.show()
        
    return segments_closed
                    


