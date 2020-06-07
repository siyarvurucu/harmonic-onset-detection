import sys, os
import asyncio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyaudio
import wave

import tkinter as tk
from tkinter import *
from tkinter import filedialog as tkFileDialog
from tkinter import messagebox as tkMessageBox

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backend_bases import Event

def quickPlayer(soundfile,data,stamps,names,h=128,onlyValues=False):
    """
    onlyValues: If  provided time stamps does not hae IDs or names, this should be True. Manually annotated files will have IDs and names for the stamps  . 
    
    soundfile: sound file to be played
    data: data that shown on the player plot
    stamps: lines will be drawed on these points on the data
    h: hopsize
    
    size(data) * h = size(sound)      If you pass audio samples as data, h (hop size) should be 1.
    """
    
    wf = wave.open(soundfile,'rb')
    parent = Tk() 

    chunk_size = 2048
    sem = asyncio.Semaphore()
    play_mode = tk.BooleanVar(parent, False)
    
    fig = plt.figure(figsize=(18,7), dpi=100)
    mainplot = fig.add_subplot(111)

    canvas = FigureCanvasTkAgg(fig, parent)
    canvaswidget = canvas.get_tk_widget()
    canvaswidget.grid(row=2,column=0,columnspan=5, sticky=W)

    toolbarFrame = Frame(master=parent)
    toolbarFrame.grid(row=3,column=0, columnspan=7)
    toolbar = NavigationToolbar2Tk(canvas, toolbarFrame)
    toolbar.update()

    canvas._tkcanvas.grid(row=2,column=0, columnspan = 7, sticky=W)
    global background
    background = canvas.copy_from_bbox(mainplot.bbox)
    cursor = mainplot.axvline(color="k", animated=True)
    cursor.set_xdata(0)

    for d in data:
        mainplot.plot(d)
    global x0,x1
    x0,x1 = mainplot.get_xlim() 
    
    p = pyaudio.PyAudio()


    def callbackstream(in_data, frame_count, time_info, status):
        sem.acquire()
        data = wf.readframes(frame_count)
        parent.event_generate("<<playbackmove>>",when="now")
        sem.release()
        return (data, pyaudio.paContinue)

    _callbackstream = callbackstream

    
    def initStream():
        global stream 
        stream = p.open(format=8,
                        channels=1,
                        rate=44100,
                        output=True,
                        stream_callback=_callbackstream,
                        start = True,
                        frames_per_buffer = chunk_size)
                        
    def playsound(event=None):
        if sem.locked():
            return
        if play_mode.get() == True:
                # this is broken
            global stream
            stream.close()
            play_mode.set(False)
        else:
            try:
                initStream()
                play_mode.set(True)
            except:
                print("play failed")
                
    def playbackMove(event=None): # move cursor by audio chunk size
        global x0,x1
        global background
        incr = (chunk_size)//h
        nextpos = cursor.get_xdata()+incr
        cursor.set_xdata(nextpos)
        updateCursor()
        if (x1-nextpos)<0:
            mainplot.set_xlim(x1,x1+x1-x0)
            canvas.draw()
            
            toolbar.push_current()
            background = canvas.copy_from_bbox(mainplot.bbox)
            x0,x1 =  mainplot.get_xlim()
            

    def new_release_zoom(*args,**kwargs):
        global x0,x1
        release_zoom_orig(*args, **kwargs)
        s = 'toolbar_event'   
        event = Event(s,canvas)
        canvas.callbacks.process(s, Event('toolbar_event',canvas))
        x0,x1 = mainplot.get_xlim()
        
    def new_release_pan(*args, **kwargs):
        global x0,x1
        release_pan_orig(*args, **kwargs)
        s = 'toolbar_event'   
        event = Event(s,canvas)
        canvas.callbacks.process(s, Event('toolbar_event',canvas))
        x0,x1 = mainplot.get_xlim()

    def new_update_view(*args, **kwargs):
        global x0,x1
        _update_view_orig(*args, **kwargs)
        s = 'toolbar_event'  
        event = Event(s,canvas)
        canvas.callbacks.process(s, Event('toolbar_event',canvas)) 
        x0,x1 = mainplot.get_xlim()

    def handle_toolbar(event):
        global x0,x1
        canvas.draw()
        global background
        background = canvas.copy_from_bbox(mainplot.bbox)          

    def onclick(event):
        if (toolbar._active == 'ZOOM' or toolbar._active == 'PAN'):
            pass
        else:
            cursor.set_xdata(event.xdata)
            wf.setpos(int(event.xdata*h))
            updateCursor()
            
    def updateCursor():
        canvas.restore_region(background)
        mainplot.draw_artist(cursor)
        canvas.blit(mainplot.bbox)
        
    parent.bind("<space>",playsound)
    parent.bind("<<playbackmove>>",playbackMove)
    
    release_zoom_orig = toolbar.release_zoom
    toolbar.release_zoom = new_release_zoom
    release_pan_orig = toolbar.release_pan
    toolbar.release_pan = new_release_pan
    _update_view_orig = toolbar._update_view
    toolbar._update_view = new_update_view

    canvas.mpl_connect('toolbar_event', handle_toolbar)
    cid1 = canvas.mpl_connect("button_press_event", onclick)
    

    canvas.draw()
    
    ymax = max(data[0])
    
    colors = ['c','m','y','r','#FFBD33','#924A03','#D00000','#D000D0','#6800D0','#095549','b','r','r']
    if onlyValues:
        for i in range(len(stamps)):
            mainplot.draw_artist(mainplot.vlines(x=stamps[i],color=colors[i],ymin=-ymax,ymax=ymax))
    else:   
        for i in range(len(stamps)):
            for j in stamps[i]:
                mainplot.draw_artist(mainplot.axvline(x=j[1],color=colors[i]))
                mainplot.text(j[1]+2, (0.9*ymax)+(i*0.02), names[i] + j[0], bbox=dict(fill=False, edgecolor=colors[i], linewidth=1))
    canvas.draw()
    background = canvas.copy_from_bbox(mainplot.bbox)
    
    parent.mainloop()
      
    for func in [stream.close,
             wf.close,
             p.terminate]:
        try:
            func()
        except:
            pass

    parent.destroy() 
    
    return