from os import listdir
from os.path import isfile, join, abspath, dirname
import sys

import PyPDF2 as pypdf
#import pdftotree
#from pdfminer import pypdf2xml

BASE_DIR = abspath(dirname(__file__) )
def get_files_as_array():
    TDS_PATH = BASE_DIR + '/data/tds/'
    return [f for f in listdir(TDS_PATH) if isfile(join(TDS_PATH, f))], TDS_PATH

def findInDict(needle, haystack):
    for key in haystack.keys():
        try:
            value=haystack[key]
        except:
            continue
        if key==needle:
            return value
        if isinstance(value,dict):            
            x=findInDict(needle,value)            
            if x is not None:
                return x
def get_raw_xml_pdf_data(filepath):
    with open(filepath,'rb') as f:
        pdf = pypdf.PdfFileReader(f)

        # XFA
        xfa = findInDict('/XFA',pdf.resolvedObjects)
        xml = xfa[7].getObject().getData()


    return xml

def get_raw_acroforms_pdf_data(filepath):
    with open(filepath,'rb') as f:
        pdf = pypdf.PdfFileReader(f)

        # Acroforms
        acroforms = pdf.getFormTextFields()

    return acroforms

def get_html(filepath):
    if sys.version_info > (3, 0):
        from io import StringIO
    else:
        from io import BytesIO as StringIO
    from pdfminer.layout import LAParams
    from pdfminer.high_level import extract_text_to_fp
    output_string = StringIO()
    with open(filepath, 'rb') as f:
        a = extract_text_to_fp(f, output_string, laparams=LAParams(),output_type='html', codec=None)
        b = 2

        
def get_general_data_of_pdf(filepath):
    with open(filepath,'rb') as f:
        pdf = pypdf.PdfFileReader(f)
        info = pdf.getDocumentInfo()
        number_of_pages = pdf.getNumPages()


        for page in pdf.pages:
            text = page.extractText()
            a = text.split('\n')
            b = 1

    author = info.author
    creator = info.creator
    producer = info.producer
    subject = info.subject
    title = info.title

    return info

def main():
    filenames, tds_path = get_files_as_array()
    for filename in filenames:
        print(tds_path + filename)
        #a = pdftotree.parse(filename, html_path=None, model_type=None, model_path=None, visualize=False)
        #a = pypdf2xml(tds_path + filename)
        info = get_html(tds_path + filename)

if __name__ == '__main__':
    main()


'''

pdf2txt.py /Users/alpha/Documents/GitHub/tds_db_generator/src/data/tds/Autofroth 1402-1 Resin & Autofroth 10000A Isocyanate.pdf

'''