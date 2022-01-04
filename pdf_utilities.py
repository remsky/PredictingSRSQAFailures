
import io
import os
import re
import bisect
import glob 
import pydicom 
import pdfminer
import pandas as pd
import numpy as np

from io import StringIO
from io import BytesIO

from fuzzywuzzy import fuzz
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdf2image import convert_from_path
from PIL import Image
from progressbar import ProgressBar

# OCR REQUIRES FULL TESSERACT INSTALLATION, and must be added to path:
# see https://pypi.org/project/pytesseract/  -- 'Installation'
# Can skip this (currently unused) but if result was 'printed to pdf'
# instead of downloaded at Octavius terminal, requires a visual read and/or manual check (rare)
import pytesseract

# General utilities for directory crawling and simple PDF mining 
def pdf_to_text(pdfname):
    # via https://gist.github.com/jmcarp/7105045
    
    # PDFMiner boilerplate
    rsrcmgr = PDFResourceManager()
    sio = StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, sio, codec=codec, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    
    # Extract text 
    page_list = [] #
    fp = open(pdfname, 'rb')
    for page in PDFPage.get_pages(fp):
        page_list.append(page) # 
        #interpreter.process_page(page)
    interpreter.process_page(page_list[0]) #only need 1st page
    fp.close()

    # Get text from StringIO
    text = sio.getvalue()

    # Cleanup
    device.close()
    sio.close()
    return text

def split(delimiters, string, maxsplit=0):
    regexPattern = '|'.join(map(re.escape, delimiters))
    return re.split(regexPattern, string, maxsplit)

def printPath(rootFolder):
    fname = rootFolder.split(os.sep)[-1]
    root_levels = rootFolder.count(os.sep)
    output = []
    lastroot = ""
    cnt = 0
    # os.walk treats dirs breadth-first, but files depth-first
    for root, dirs, files in os.walk(rootFolder):
        for fi in files:
            if ".pdf" in fi:
                cnt += 1
                if cnt % 250 == 0:
                    print(f"            :  {cnt} filepaths collected")
                updir = root.split(os.sep)[:-1]
                time = os.path.getmtime(os.path.join(root,fi))
                updir = os.path.join(*updir)
                output.append([root.split(os.sep)[-2],os.path.join(root,fi),updir,root,time])
 
    return output                     

def select_patientID(excelfilter):
    SRSpatient = pd.read_excel(excelfilter,skiprows=3)
    return SRSpatient
        
def flatten(l, ltypes=(list, tuple)):
    # via https://stackoverflow.com/questions/716477/join-list-of-lists-in-python
    ltype = type(l)
    l = list(l)
    i = 0
    while i < len(l):
        while isinstance(l[i], ltypes):
            if not l[i]:
                l.pop(i)
                i -= 1
                break
            else:
                l[i:i + 1] = l[i]
        i += 1
    return ltype(l)

def organization_by_path_and_time(list_of_paths):
    sorted_list_of_paths = sorted(list_of_paths, key=lambda x: ((x[3]), x[4])) #sorting first by folder, then time 
    return sorted_list_of_paths

def glob_plan_dataset(directory):
    plan_dataset_filepath = glob.glob(os.path.join(directory, 'RP*dcm')) 
    if len(plan_dataset_filepath) < 1:
            plan_dataset_filepath = glob.glob(os.path.join(*[directory,"Calculated"], 'RP*dcm'))
            if len(plan_dataset_filepath) < 1:
                for path, directories, files in os.walk(directory):
                    if file in files:
                        print(f" Additional walk used for : {path}")
                        print(directories)
                        plan_dataset_filepath = glob.glob(os.path.join(path, 'RP*dcm'))
            
            
    try:
        filepath_res = plan_dataset_filepath[0]    
    except Exception as e:
        print(f"DICOM capture failure in {directory}")
        return e
    return filepath_res
    
def glob_dose_datasets(directory):
    plan_dataset_filepath = glob.glob(os.path.join(directory, 'RD*dcm')) 
    if len(plan_dataset_filepath) < 1:
            plan_dataset_filepath = glob.glob(os.path.join(*[directory,"Calculated"], 'RD*dcm'))
            if len(plan_dataset_filepath) < 1:
                for path, directories, files in os.walk(directory):
                    if file in files:
                        print(f" Additional walk used for : {path}")
                        print(directories)
                        plan_dataset_filepath = glob.glob(os.path.join(path, 'RD*dcm'))
            
            
    try:
        filepath_list = plan_dataset_filepath    
    except Exception as e:
        print(f"RTDose capture failure in {directory}")
        return e
    return filepath_list

def mine_multipath_pdf(pathlist,verbose=False): 
    collect = []
    count = 0
    for path in pathlist:
        count += 1
        #print(os.path.join(folderpath, filename))
        collect.append(mine_single_octa(path[1]))
        if count % 10 == 0 and verbose:
            print(count + "# of pdfs' mined") 

    return collect

def mine_many_octa(folderpath,verbose=False):  
    collect = []
    count = 0
    for filename in os.listdir(folderpath):
        if filename.endswith(".pdf"):
            count += 1
            #print(os.path.join(folderpath, filename))
            collect.append(mine_single_octa(folderpath + filename),filename)
            if count % 10 == 0 and verbose:
                print(count + "# of pdfs' mined") 
            continue
        else:
            return collect
    return collect 

def crop_pdf_for_ocr(image_of_pdf):
    #unused
    imag_DTA = image_of_pdf.crop((200,387,275,425))
    imag_DD =  image_of_pdf.crop((200,420,275,450))
    imag_gamma =  image_of_pdf.crop((1365,735,1450,805))
    return [imag_DTA,imag_DD,imag_gamma]

# Buggy, Optical Character Recognition (OCR) via Google Tesseract
# Currently the only method explored for 'Print to PDF' exported Octavius results
# unused in lieu of manual entry
def pdf_ocr_scrape(pdfpath):
    #unused
    results = []
    page = convert_from_path(pdfpath, 290)[0]
    cropped_segments = crop_pdf_for_ocr(page)
    for segment in cropped_segments:
        text = pytesseract.image_to_string(segment,config="digits")
        results.append(text)

    return results
    
def clean_text(stringin,pops,firstthree=False,patientid=False):
    collection = []
    delimiters = ";", "\n","'","=",'"',"'"
    split_string = split(delimiters, stringin)
    split_stringfilt = list(filter(lambda a: len(a) > 0 , split_string)) # applying filter for empty items
    clean_string = [x.strip(' ') for x in split_stringfilt]  
    if patientid == False:
        for i in pops:
            if firstthree==True: #converts strings into floats for DTA and DD
                try:  # hardcoded but validated. occasional extra character thows off index.
                    clean_string[i] = float(clean_string[i][0:4])
                except:
                    clean_string[i] = float(clean_string[i-1][0:4])
            collection.append(clean_string[i])
            
    elif patientid == True:
        patientid = list(filter(lambda x: re.search('[a-zA-Z]{1}\d{6}', x), clean_string))
        collection.append(patientid)
        
    return collection 

def mine_single_octa(pdfpath):
 
    singleCollection = []
    raw_text = pdf_to_text(pdfpath)
    
    if len(raw_text)<2:
        try:
            singleCollection.append(pdf_ocr_scrape(pdfpath))
            if None in singleCollection:
                singleCollection = ["Not an ID",0,0]
            return singleCollection
        except:
            pass
    try:
        a,b = raw_text.split('Volume',1)
        c,d = b.split("Statistics",1) # grab passing rate here from D
        # hardcoded
        singleCollection.append(clean_text(a,[1],False,True))
        #ID
        singleCollection.append(clean_text(b,[1,2],True))
    
        singleCollection.append(clean_text(d,[10],True))
    except:
#       singleCollection.append(pdf_ocr_scrape(pdfpath))
        #print("Error in " + pdfpath)
        #print(f"RAW: {raw_text}")
        singleCollection.extend(["Not an ID",0,0])
        
        return singleCollection
    
    return singleCollection 
            
            

def glob_arc_pdfs(directory):
    arc_pdf_filepaths = glob.glob(os.path.join(directory, '*pdf'))
    #remove combined beam reports, if any
    arc_pdf_filepaths = list(filter(lambda x: not re.search('(?i)all|(?i)total|(?i)combined', x), arc_pdf_filepaths)) #(?i) is case-insensitive 
    return arc_pdf_filepaths


def beam_sequence_init(plan_dataset):
    # creates lookup between order of beam numbers and actual beam number in dicom (as they are occasionally misaligned) 
    defined_beam_number_dict = {}
    beam_sequence = plan_dataset.BeamSequence
    num_of_beams = len(beam_sequence)
    for i in range(0,num_of_beams):
        ds_defined_number = plan_dataset.FractionGroupSequence[0].ReferencedBeamSequence[i].ReferencedBeamNumber
        defined_beam_number_dict[int(ds_defined_number)] = i   # {sequence list index:dataset number}

    return defined_beam_number_dict


# This is the heavy lifter, collects pdfs and associated DICOM files together, along with confidence of match. 
# Confidence < 30 is not worth keeping. Above that, regular spot-checking has found no mis-matches. 
def pdf_dicom_match_collector(raw_directory_string,filter_sheet, RTDOSEcollect = False, verbose=True,testmode=False):
    print("Initializing: Collecting PDF filepaths")
    pathlist = printPath(raw_directory_string)

    print("Initializing: Sorting PDF filepaths")
    pathlist = organization_by_path_and_time(pathlist)
    # create unique list of folder paths here. 
    count = 0  # count of pds crawled
    count_success = 0 # count of pdf's mined.
    count_not_in_filter = 0 # count of pdfs mapped which are not in the filter
    delimiters = "\\"
    
    collecter = []
    pathcollect = []
    name_number_collect = []
    failed_collect_list = []
    dose_dicom_list = []

    #PSEUDO:
    ## a path is returned for each pdf found
    ## use that path truncuated up one to find folder, enter into a set to remove redundancies 
    ## create list of all pdf's in each unique folder. store folder name and path for each as safety
    ## append all pdf scrape in that folder as sublists
    ## then find 'best match' to ds info (from up another folder) instead of absolute match. 
    ### can rank confidence on token_sort_ratio/regex between pdfname/dsname, 
    ### if exact regex match (common), confidence pushes to 100
    ## then move on to next folder 
    
    print("Initializing: Creating PDF pathlist set")

    folder_set = set()
    for path in pathlist:
        if ("all" in path[1].lower()) or ("total" in path[1].lower()):
            continue
        folder_set.add(path[3])
        
    print("Working     : Capturing PDF associated DICOMs")
    print("Working     : Mining PDF's for Octavius Result")
    pbar = ProgressBar() 
    for folderpath in pbar(folder_set):
        ds_dir = os.path.dirname(folderpath)
        try:
            plan_dicom = glob_plan_dataset(ds_dir)
            if RTDOSEcollect:
                dose_dicom_list = glob_dose_datasets(ds_dir)
                
            ds = pydicom.dcmread(plan_dicom)
        except:
            failed_collect_list.append(folderpath)
            continue 

        beam_sequence_numbering = beam_sequence_init(ds)
        pdf_list_in_folder = glob_arc_pdfs(folderpath)
        ds_collect = []
        
        for beam in ds.BeamSequence:
                ds_collect.append([beam.BeamName,beam.BeamNumber])
      
        for target_pdf in pdf_list_in_folder:
            count += 1
            confidences = {}
            name_dict = {}
            pdf_file_name = target_pdf.split("\\")[-1].split(".")[0]
            flagger = " "
            regex_match = False
            
            for dsname in ds_collect:
                # create confidence critera
                ## fuzzy matching ratios:
                token_set_res = fuzz.token_sort_ratio(dsname[0],pdf_file_name)
                
                ## REGEX: Beam/Arc # matching filename criteria. 
                pdf_file_name_squeeze = pdf_file_name.replace(" ", "") #squeeze before regex to remove possible whitespace 
        
                search_object = re.search('((?i)beam|(i?)arc)(\d)', pdf_file_name_squeeze)
                if search_object:        
                    if int(search_object.group(3)) == int(dsname[1]):
                        ## if regex perfect match, push confidence to 100 
                        regex_match = True
                        token_set_res = 100
                
                        
                    
                confidences[dsname[1]] = token_set_res
                name_dict[dsname[1]] = dsname[0]

            key_max = max(confidences.keys(), key=(lambda k: confidences[k]))
            key_name = dsname
            beam_index = beam_sequence_numbering[key_max]  # use maximum confidence match 
            
            dose_dicom = "FAILED" #init
            for i in range(0,len(dose_dicom_list)):
                      
                match_beam = re.search('BEAM_(\d+)', dose_dicom_list[i])
                if match_beam:
                    match_beam = match_beam.group(0).split("_")[1]
                    if (int(match_beam) ==  int(key_max)):
                        dose_dicom = dose_dicom_list[i]
                    else:
                        pass

            unprocessed_patient = mine_single_octa(target_pdf) #indexed directly
            unprocessed_patient = flatten(unprocessed_patient,(list)) #Result, C#, Gamma results. 

            failed_capture = [unprocessed_patient,target_pdf,0,0]
       
            try:
                if filter_sheet['Patient ID'].str.contains(unprocessed_patient[0]).any():#if ID in Excel sheet of SRS plans              
                    pathcollectStr = target_pdf.split("\\")[-1]
                    dsnamer = name_dict[key_max]
                    collecter.append(flatten([unprocessed_patient,plan_dicom,beam_index,dsnamer,confidences[key_max],pathcollectStr,dose_dicom],(list)))  #flatting multi-d list to one
                    count_success += 1
                else:
                    count_not_in_filter += 1

                # if ((testmode == True) & (count == 50)):
                    # print(f"Test mode complete. Count of {count} crawled")
                    # return collecter
                
            except IndexError as Ind:
                print("List Index beam {} mis-match for file\n | |  {} \n | |  {} \n*---*".format(target_pdf,key_max,traceback.print_exc()))
                collecter.append(flatten(failed_capture,(list))) 
                pathcollect= [] # have to reset list here as well
                continue
            except AttributeError as Att:
                print(f"No such attribute error in {target_pdf}")
                collecter.append(flatten(failed_capture,(list))) 
                pathcollect= [] # have to reset list here as well
                continue
            except TypeError as Typ:
                print(f"Type Error in : {target_pdf}")
                print(f"Unprocessed data mismatch in: {unprocessed_patient}") 
                collecter.append(flatten(failed_capture,(list))) 
                pathcollect= [] # have to reset list here as well
                continue            

    print(f"{count_success} PDF's successfully mined")
    print(f"{len(failed_collect_list)} failed to capture")
    print(f"{count_not_in_filter} filtered out, non-SRS")
    print("DICOM capture failures in: ")
    for i in range(0,len(failed_collect_list)):
        print(f" {failed_collect_list[i]}")
    print("\n * DICOM and PDF collection complete *")
    return collecter     