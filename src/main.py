import os
import sys

import pandas as pd

from pdf_reader import Pdf_reader, BASE_DIR, EXCLUDE_CHARACTERS
from general_pkg import string_comparison
from bs4 import BeautifulSoup


# Files
def get_files_as_array(datatype: str='pdf') -> list:
    TDS_PATH = BASE_DIR + '/data/tds_pdf/'
    return [f for f in os.listdir(TDS_PATH) if f.endswith(datatype) if os.path.isfile(os.path.join(TDS_PATH, f))], TDS_PATH

class Analyze_Html(object):
    def __init__(self, pdf_reader): 
        self.__pdf_reader = pdf_reader

    @staticmethod
    def __save_html_as_file(html_text, filename, delete_=False):
        filename = filename.replace('tds_pdf','tds_html').replace('.pdf','.html')
        with open(filename, 'w') as f:
            f.write(html_text)

        if delete_:
            os.remove(filename)

    def __get_soup(self, filename):
        soup = None
        html_text = None
        with open(filename, 'rb') as f:
            html_text = self.__pdf_reader.extract_html_from_pdf_with_pdf_miner(f)
            soup = BeautifulSoup(html_text,'html.parser')

        return soup, html_text

    def get_header_dict_from_html(self, filename: str) -> tuple:
        soup, html_text = self.__get_soup(filename)
        header_analyzer = self.__Analyze_Header()
        headers, header_dict = self.__get_info_from_soup(soup, header_analyzer)

        self.__save_html_as_file(html_text, filename, delete_=True)

        return headers, header_dict

    def get_footer_dict_from_html(self, filename: str) -> tuple:
        soup, html_text = self.__get_soup(filename)
        footer_analyzer = self.__Analyze_Footer()
        footers, footer_dict = self.__get_info_from_soup(soup, footer_analyzer)

        self.__save_html_as_file(html_text, filename, delete_=True)

        return footers, footer_dict

    def __get_info_from_soup(self, soup, analyzer):
        info_set = set()
        info_dict = dict()
        
        div_style_tag = 'position:absolute; border: textbox 1px solid; writing-mode:lr-tb;'
        div_containers = soup.find_all('div',style=lambda value: value and div_style_tag in value)
        
        page = 1
        page_found = False
        for div_container in div_containers:
            info_set, info_dict, page, page_found = analyzer.get_info(div_container, info_set, info_dict, page, page_found)
            #get_values(div_container)
            if page_found:
                page_found = False
            
        return info_set, info_dict


    class __Analyze_Header(object):
        @staticmethod
        def get_info(div_container, info_set, info_dict, page, page_found):
            headers_per_page = set()

            span_headers = div_container.find_all('span', {'style':'font-family: BookmanOldStyle-Bold; font-size:10px'})
            for span_header in span_headers:
                header = span_header.text
                header = header.replace('\n','').strip()
                if header not in EXCLUDE_CHARACTERS:
                    info_set.add(header)
                    headers_per_page.add(header)
                    page_found = True

            if page_found:
                page += 1
                info_dict[page] = headers_per_page
                headers_per_page = None
                headers_per_page = set()

            return info_set, info_dict, page, page_found

    class __Analyze_Footer(object):
        @staticmethod
        def get_info(div_container, info_set, info_dict, page, page_found):
            footer_per_page = set()

            span_footers = div_container.find_all('span', {'style':'font-family: BookmanOldStyle; font-size:7px'})#
            span_footers += div_container.find_all('span', {'style':'font-family: ArialMT; font-size:6px'})
            # span_footers += div_container.find_all('span', {'style':'font-family: ArialRoundedMTBold; font-size:7px'})
            # span_footers += div_container.find_all('span', {'style':'font-family: ArialMT; font-size:7px'})
            # span_footers += div_container.find_all('span', {'style':'font-family: ArialMT; font-size:9px'})
            # span_footers += div_container.find_all('span', {'style':'font-family: Arial-BoldMT; font-size:6px'})
            # span_footers += div_container.find_all('span', {'style':'font-family: ArialRoundedMTBold; font-size:6px'})
            for span_footer in span_footers:
                footer = span_footer.text
                footer = footer.replace('\n','').strip()
                if footer not in EXCLUDE_CHARACTERS and len(footer) > 5:
                    info_set.add(footer)
                    footer_per_page.add(footer)
                    page_found = True

            if page_found:
                page += 1
                info_dict[page] = footer_per_page
                footer_per_page = None
                footer_per_page = set()

            return info_set, info_dict, page, page_found

class Analyze_Text(object):
    def __init__(self):
        self.__nlp_max_value = 90
        self.__footer_keyword = 'END'

    def __get_info_dict(self, info_set: list, pdf_text: dict) -> dict:
        info_dict = dict()
        info_dict_ = dict()
        print('\n')
        for page_number, page in enumerate(pdf_text):
            info_dict[page_number + 1] = dict()
            info_dict_[page_number + 1] = dict()
            for info in info_set:
                for index, line in enumerate(page):
                    # if 'Important' in line and 'Important' in info:
                    #     a = 1
                    match, matching_value = string_comparison(info, line, max_value=self.__nlp_max_value)
                    if match:
                        print(f'{matching_value}\t{info[:15]}\t\t{line[:20]}')
                        info_dict[page_number + 1][index] = line
            for key, value in sorted(info_dict[page_number + 1].items()):
                info_dict_[page_number + 1][key] = value

        return info_dict_



    def get_pdf_dict(self, headers: set, footers: set, pdf_text: dict, pdf_text_unsorted: dict) -> dict:
        headers_dict = self.__get_info_dict(headers, pdf_text)
        footer_dict = self.__get_info_dict(footers, pdf_text)

        # add footer to info_dict
        info_dict = headers_dict.copy()
        for page_number in info_dict.keys():
            key = list(footer_dict[page_number].keys())[0]
            # value = footer_dict[page_number][key]
            info_dict[page_number][key] = self.__footer_keyword#value


        re = dict()
        for page_number in info_dict.keys():
            for index in range(len(info_dict[page_number])):
                header_line = list(info_dict[page_number])[index]
                header = info_dict[page_number][header_line]
                if header == self.__footer_keyword:
                    break
                next_header_line = list(info_dict[page_number])[index + 1]
                re[header] = list()
                for line in range(header_line, next_header_line - 1):
                    value = pdf_text[page_number - 1][line + 1]
                    re[header].append(value)

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
        html_analyzer = Analyze_Html(pdf_reader)
        headers, header_dict = html_analyzer.get_header_dict_from_html(tds_path + filename)
        footers, footer_dict = html_analyzer.get_footer_dict_from_html(tds_path + filename)

        # get text
        text_tesseract = pdf_reader.get_text_from_pdf(tds_path, filename, method=4, output_type='text', dpi=500)
        text_tesseract = pdf_reader.clean_data(text_tesseract)
        text_PyMuPDF = pdf_reader.get_text_from_pdf(tds_path, filename, method=2, output_type='text')
        text_PyMuPDF = pdf_reader.clean_data(text_PyMuPDF)
        
        analyzer = Analyze_Text()
        pdf_dict = analyzer.get_pdf_dict(headers, footers, text_tesseract, text_PyMuPDF)

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