import os 
import pathlib
import pandas as pd
import numpy as np
import tgt
import re


def read_tgt(session):
    filepath = 'transcriptions_annotations/' + session
    tg_path = filepath + '/' + session + '.TextGrid'
    tg = tgt.read_textgrid(tg_path) # my text grid files
    print(tg_path)
    return tg 

def load_session_annotation(doc_no):
    tg = read_tgt(doc_no)
    return tg    

def get_session_start_end(tg):
    
    part = tg.get_tier_by_name('Part')
    part_intervals = part.intervals
    part.end_time
    session_start = part.intervals[0].start_time 
    session_end = part.intervals[0].end_time
    
    return { 'session_start': session_start, 'session_end': session_end } 


def listdir(folder):
    for x in os.listdir(folder):
        if x != '.DS_Store':
            yield x


def crop_out_participants(path, which_participant):
    # path = "videos"
    output_folder = pathlib.Path('output')
    output_path = path / output_folder

    if which_participant == 'a':
        crop_command = str("crop=360:288:0:0")
        
    if which_participant == 'b':
        crop_command = str("crop=360:288:360:0")
    
    for i, file in enumerate(sorted(os.listdir(path))):
        if (file.endswith(".mp4")): #or .avi, .mpeg, whatever.
            command = "ffmpeg -i " + os.path.join(path, file) + " -filter:v " + crop_command + " " + os.path.join(output_path, "output_" + str(pathlib.Path(file).stem) + "_" + which_participant + ".mp4")
            os.system(command)
        else:
            continue