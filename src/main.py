from os import listdir
from os.path import isfile, join, abspath, dirname
import sys

import PyPDF2 as pypdf
from io import StringIO
from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams
from typing import BinaryIO
from bs4 import BeautifulSoup


# Files
BASE_DIR = abspath(dirname(__file__) )
def get_files_as_array(datatype=""):
    TDS_PATH = BASE_DIR + '/data/tds_pdf/'
    return [f for f in listdir(TDS_PATH) if f.endswith(datatype) if isfile(join(TDS_PATH, f))], TDS_PATH

# Pdf Extractor
# https://pdfminersix.readthedocs.io/_/downloads/en/latest/pdf/
def extract_text_from_pdf(pdf_fo: BinaryIO,output_type) -> str:
    """
    Extracts text from a PDF

    :param pdf_fo: a byte file object representing a PDF file
    :return: extracted text
    :raises pdfminer.pdftypes.PDFException: on invalid PDF
    """
    out_fo = StringIO()
    extract_text_to_fp(pdf_fo, out_fo, laparams=LAParams(), output_type=output_type, codec=None)
    return out_fo.getvalue()

def save_file_as_html(filepath):
    return __save_file_as_(filepath,'html')
      
def save_file_as_xml(filepath):
    return __save_file_as_(filepath,'xml')

def __save_file_as_(filepath,output_type):
    extracted_filename = filepath.replace('.pdf',f'.{output_type}').replace('tds_pdf',f'tds_{output_type}')
    with open(filepath, 'rb') as f:
        extracted_code = extract_text_from_pdf(f,output_type)
        with open(extracted_filename, 'w+') as h:
            h.write(extracted_code)

    return extracted_filename
      
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

# Analyze Html

EXCLUDE_HEADERS = {'',' ','\n'}

def is_nextPage(page_found, page):
    if page_found:
        page += 1
        header_dict[page] = headers_per_page
        headers_per_page = None
        headers_per_page = set()
        page_found = False

def get_headers(div_container, headers, header_dict, page, page_found):
    headers_per_page = set()

    span_headers = div_container.find_all('span', {'style':'font-family: BookmanOldStyle-Bold; font-size:10px'})
    for span_header in span_headers:
        header = span_header.text
        header = header.replace('\n','').strip()
        if header not in EXCLUDE_HEADERS:
            headers.add(header)
            headers_per_page.add(header)
            page_found = True

    if page_found:
        page += 1
        header_dict[page] = headers_per_page
        headers_per_page = None
        headers_per_page = set()

    return headers, header_dict, page, page_found

def get_values(div_container):
    span_parameters = div_container.find_all('span', {'style':'font-family: BookmanOldStyle-Bold; font-size:10px'})
    if span_parameters != []:
        a = 1
    for span_parameter in span_parameters:
        parameter = span_parameter.text
        parameter = parameter.replace('\n','').strip()
        if parameter not in EXCLUDE_HEADERS:
            a = 1

def get_headers_from_soup(soup):
    headers = set()
    header_dict = dict()
    
    div_header_style_tag = 'position:absolute; border: textbox 1px solid; writing-mode:lr-tb;'
    div_header_containers = soup.find_all('div',style=lambda value: value and div_header_style_tag in value)
    
    page = 1
    page_found = False
    for div_header_container in div_header_containers:
        headers, header_dict, page, page_found = get_headers(div_header_container, headers, header_dict, page, page_found)
        #get_values(div_header_container)
        if page_found:

            page_found = False
        
    return headers, header_dict

def get_dict_from_html(filename):
    headers = set()
    header_dict = dict()
    with open(filename, 'rb') as f:
        soup = BeautifulSoup(f,'html.parser')
        headers, header_dict = get_headers_from_soup(soup)

    return headers, header_dict

def main():
    filenames, tds_path = get_files_as_array(datatype=".pdf")
    for filename in filenames:
        print(tds_path + filename)
        html_filename = save_file_as_html(tds_path + filename)
        # xml_filename = save_file_as_xml(tds_path + filename)
        headers, header_dict = get_dict_from_html(html_filename)
        a = 1

if __name__ == '__main__':
    main()


'''

pdf2txt.py /Users/alpha/Documents/GitHub/tds_db_generator/src/data/tds/Autofroth 1402-1 Resin & Autofroth 10000A Isocyanate.pdf

'''