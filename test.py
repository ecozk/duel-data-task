import os
import glob
import tgt
import sys
import pandas as pd
from re import match, sub, findall, finditer

root_dir = '../'
word_level_timing = root_dir+'word_level_timings'
motion_label = root_dir+'motion_labels' 
original_annotation = root_dir + 'transcriptions_annotations'
def clean_utt(utt, literal=False):
    if not literal:
        #replace variants, partial and misspoken words with standard spelling
        utt = sub("""<[vpm]="(.+?)">.+?</[vpm]>""", lambda m:m.group(1), utt)
        #remove fillers like "{F aehm}" entirely
        utt = sub("""{.*?}""", "", utt)
        
        #TO DO: resolve complex replacements like "(der + der) + die) Katze"
        
    else:
        #remove brackets from fillers, i.e. "{F aehm}" becomes "aehm"
        utt = sub("""{(.*?)}""",lambda m:m.group(1),utt)
    #remove all remaining xml-style tags    
    utt = sub("""<.*?>""","",utt)
    #remove open tags at the end of an utterance (can be removed once problems with the TextGrids are fixed)
    utt = sub("""<.*$""","",utt)
    #remove all remaining punctuation and brackets
    utt = sub("""[\.:;,\(\)\+\$]""","",utt)
    #remove whitespace at the beginning and end of an utterance
    utt = utt.strip()
    #replace any amount of whitespace with a single space
    utt = sub("""\s+"""," ",utt)
    return utt



def get_all_textgrid_files(path):
    filenames = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".TextGrid"):
                filenames.append(os.path.join(root, file))
    return filenames
if __name__ == '__main__':
    annotations = get_all_textgrid_files(original_annotation)
    print(annotations)
    for index, f in enumerate(annotations):
        clear_text = list()
        textgrid = tgt.read_textgrid(f)
        tier_names = textgrid.get_tier_names()
        print("*"*50)
        print(tier_names)
        print("*"*50)
        for names in tier_names:
            text_tier = textgrid.get_tier_by_name(names)
            anno = text_tier.annotations
            for annotation in anno:
                clear_text.append((f, names, annotation.text, clean_utt(annotation.text), annotation.start_time, annotation.end_time))
                print(f"Text :{annotation.text}, clear Text: {clean_utt(annotation.text)}, Start Time: {annotation.start_time}, End Time: {annotation.end_time}")
        print("-"*50)
