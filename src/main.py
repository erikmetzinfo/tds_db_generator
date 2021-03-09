import os
import sys

import pandas as pd

from pdf_reader import Pdf_reader
from bs4 import BeautifulSoup


# Files
BASE_DIR = os.path.abspath(os.path.dirname(__file__) )
def get_files_as_array(datatype: str='pdf') -> list:
    TDS_PATH = BASE_DIR + '/data/tds_pdf/'
    return [f for f in os.listdir(TDS_PATH) if f.endswith(datatype) if os.path.isfile(os.path.join(TDS_PATH, f))], TDS_PATH

class Analyze_Headers(object):
    def __init__(self):
        self.__BASE_DIR = os.path.abspath(os.path.dirname(__file__))

    @staticmethod
    def __get_headers(div_container, headers, header_dict, page, page_found):
            headers_per_page = set()

            span_headers = div_container.find_all('span', {'style':'font-family: BookmanOldStyle-Bold; font-size:10px'})
            for span_header in span_headers:
                header = span_header.text
                header = header.replace('\n','').strip()
                if header not in EXCLUDE_CHARACTERS:
                    headers.add(header)
                    headers_per_page.add(header)
                    page_found = True

            if page_found:
                page += 1
                header_dict[page] = headers_per_page
                headers_per_page = None
                headers_per_page = set()

            return headers, header_dict, page, page_found

    def __get_headers_from_soup(self, soup):
            headers = set()
            header_dict = dict()
            
            div_header_style_tag = 'position:absolute; border: textbox 1px solid; writing-mode:lr-tb;'
            div_header_containers = soup.find_all('div',style=lambda value: value and div_header_style_tag in value)
            
            page = 1
            page_found = False
            for div_header_container in div_header_containers:
                headers, header_dict, page, page_found = self.__get_headers(div_header_container, headers, header_dict, page, page_found)
                #get_values(div_header_container)
                if page_found:

                    page_found = False
                
            return headers, header_dict

    def __get_dict_from_html(self, filename: str) -> tuple:
        headers = set()
        header_dict = dict()
        with open(filename, 'rb') as f:
            soup = BeautifulSoup(f,'html.parser')
            headers, header_dict = self.__get_headers_from_soup(soup)

        return headers, header_dict

    def get_dict_from_html(self, html_set: set) -> tuple:
        re = None
        html_text = ""
        html_filepath = f'{self.__BASE_DIR}/temp.html'
        for html_text in html_set:
            with open(html_filepath, 'w') as f:
                f.write(html_text)
                re = self.__get_dict_from_html(html_filepath)
                os.remove(html_filepath)
        return re


class Analyze_Text(object):
    def __init__(self):
        self.__BASE_DIR = os.path.abspath(os.path.dirname(__file__))

    def get_pdf_dict(self, headers: list, pdf_text: dict, pdf_text_unsorted: dict) -> dict:
        re = dict()
        for header in headers:
            re[header] = list()
            header_found = False
            for index, line in enumerate(pdf_text):
                if header_found:
                    next_line = pdf_text[index + 1]
                    re[header].append(line)
                    if any(substring in next_line for substring in headers):
                        break
                elif header in line:
                    header_found = True

        return re
def main():
    pdf_reader = Pdf_reader()
    filenames, tds_path = get_files_as_array(datatype=".pdf")
    for filename in filenames:
        print(tds_path + filename)

        # get general data
        pdf_gerneral_info_dict_pypdf = pdf_reader.get_general_data_of_pdf_with_pypdf(tds_path + filename)
        pdf_gerneral_info_dict_PyMuPDF = pdf_reader.get_general_data_of_pdf_with_PyMuPDF(tds_path + filename)

        # get text
        # text_pdf_miner = pdf_reader.get_text_from_pdf(tds_path, filename, method=1, output_type='text')
        # text_pdf_miner = pdf_reader.clean_data(text_pdf_miner)
        
        # text_PyPDF2 = pdf_reader.get_text_from_pdf(tds_path, filename, method=3, output_type='text')
        # text_PyPDF2 = pdf_reader.clean_data(text_PyPDF2)
        
        # get headers
        html_set1 = pdf_reader.get_text_from_pdf(tds_path, filename, method=2, output_type='html')
        html_set2 = pdf_reader.get_text_from_pdf(tds_path, filename, method=1, output_type='html')
        html_set3 = pdf_reader.get_text_from_pdf(tds_path, filename, method=3, output_type='html')
        headers = Analyze_Headers()
        headers, header_dict = headers.get_dict_from_html(html_set1)

        # get text
        text_tesseract = pdf_reader.get_text_from_pdf(tds_path, filename, method=4, output_type='text', dpi=500)
        text_tesseract = pdf_reader.clean_data(text_tesseract)
        text_PyMuPDF = pdf_reader.get_text_from_pdf(tds_path, filename, method=2, output_type='text')
        text_PyMuPDF = pdf_reader.clean_data(text_PyMuPDF)
        
        analyzer = Analyze_Text()
        pdf_dict = analyzer.get_pdf_dict(headers, text_tesseract, text_PyMuPDF)

    a = 999

"""
def main():
    pdf_reader = Pdf_Reader()
    filenames, tds_path = get_files_as_array(datatype=".pdf")
    for filename in filenames:
        print(tds_path + filename)
        html_filename = save_file_as_html(tds_path + filename)
        headers, header_dict = get_dict_from_html(html_filename)
        # xml_filename = save_file_as_xml(tds_path + filename)
        # img_folder = save_file_as_images(tds_path, filename)
        img_filefolder, text_list = get_file_as_text_PyMuPDF(tds_path, filename)

        # method 1
        extracted_code = get_file_as_text(tds_path + filename)
        rows1 = delete_no_info_rows(extracted_code.split('\n'))
        header_rows1 = get_header_rows(rows1, headers)
        result_dict1 = get_result_dict(header_rows1, rows1, headers)

        # method 2
        rows2 = get_text_from_pdf(tds_path + filename)
        rows2 = delete_no_info_rows(rows2)
        header_rows2 = get_header_rows(rows2, headers)
        result_dict2 = get_result_dict(header_rows2, rows2, headers)
        a = 1
"""

if __name__ == '__main__':
    main()


'''

pdf2txt.py /Users/alpha/Documents/GitHub/tds_db_generator/src/data/tds/Autofroth 1402-1 Resin & Autofroth 10000A Isocyanate.pdf

'''