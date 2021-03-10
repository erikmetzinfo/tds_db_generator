import os
import sys
import shutil
import pickle

import pandas as pd
from bs4 import BeautifulSoup

from pdf_reader import Pdf_reader, BASE_DIR, EXCLUDE_CHARACTERS
from general_pkg import string_comparison



# Files
def pickle_get_content(file_path):
    '''Returns the content of a pickle file.

        Args:
            file_path (str): path and name of the json file to store
                (example /Users/example_user/project_name/file_name.pickle)

        Returns:
            str: content of the pickle file as string
    '''
    with open(file_path,'rb') as f:
        try:
            content = pickle.load(f)
        except EOFError:
            content = None
        return content

def pickle_dump_content(file_path,content):
    '''Write the content into a pickle file.

        Args:
            file_path (str): path and name of the json file to store
                (example /Users/example_user/project_name/file_name.pickle)
            content (str): the text which should be stored in the pickle file
    '''
    # print('dump pickle')
    with open(file_path,'wb') as f:
        pickle.dump(content,f)

def get_files_as_array(datatype: str='pdf') -> list:
    TDS_PATH = BASE_DIR + '/data/tds_pdf/'
    return [f for f in os.listdir(TDS_PATH) if f.endswith(datatype) if os.path.isfile(os.path.join(TDS_PATH, f))], TDS_PATH

def move_files(tds_path, filename, pdf_dict):
    destination = tds_path.replace('tds_pdf','tds_pdf_done')
    dest = shutil.move(tds_path + filename, destination + filename)
    pickle_destination = tds_path + filename.replace('.pdf', '.pickle')
    pickle_dump_content(pickle_destination, pdf_dict)

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

    @staticmethod
    def __special_string_comparison(string1, string2):
        reverse_string2 = string2[::-1]
        string1_ = string1
        last_pos=0
        for c in reverse_string2:
            pos = string1_.rfind(c)
            if pos > last_pos:
                last_pos = pos
                string1_ = string1_[:pos]
        val = string1[last_pos + 1:].strip()
        return val

    def __get_info_dict(self, info_set: list, pdf_text: dict) -> dict:
        info_dict = dict()
        info_dict_ = dict()
        # print('\n')
        for page_number, page in enumerate(pdf_text):
            info_dict[page_number + 1] = dict()
            info_dict_[page_number + 1] = dict()
            for info in info_set:
                for index, line in enumerate(page):
                    # if 'Important' in line and 'Important' in info:
                    #     a = 1
                    match, matching_value = string_comparison(info, line, max_value=self.__nlp_max_value)
                    if match:
                        # print(f'{matching_value}\t{info[:15]}\t\t{line[:20]}')
                        info_dict[page_number + 1][index] = line
            for key, value in sorted(info_dict[page_number + 1].items()):
                info_dict_[page_number + 1][key] = value

        return info_dict_

    def get_pdf_dict(self, headers: set, footers: set, pdf_text: dict) -> dict:
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
    
    def get_pdf_dict_detailed(self, pdf_dict: dict, unsorted_list: list) -> dict:
        re = dict()
        for header in pdf_dict.keys():
            re[header] = dict()
            for parameter in pdf_dict[header]:

                for page in unsorted_list:
                    for line in page:
                        match, matching_value = string_comparison(parameter, line, max_value=100)
                        if match:
                            if 'Ratio' in parameter:
                                a=1
                            param = line.strip()
                            value = parameter.replace(param, '').strip()
                            value_ = self.__special_string_comparison(parameter, param)
                            
                            found_param = False
                            if parameter.startswith(param):
                                found_param = True
                            elif value_ != value:
                                if parameter.endswith(value_) and value_ != '':
                                    value = value_
                                    found_param = True
                            
                            if found_param:
                                p_in_param = False
                                for p in re[header]:
                                    match_, matching_value_ = string_comparison(param, p, max_value=95)
                                    if match_:
                                        p_in_param = True
                                        break
                                if not p_in_param:
                                    if value == '':
                                        # TODO
                                        re[header][param] = value
                                    else:
                                        re[header][param] = value
                            # else:
                            #     # special sitauation try to eliminate all spaces
                            #     param_ = line.strip().replace(' ','')
                            #     parameter_ = parameter.replace(' ','')
                            #     value = parameter_.replace(param_, '').strip()
                            #     if parameter_.startswith(param_):
                            #         a=1
                                

        detailed_dict = dict()
        for header in pdf_dict.keys():
            detailed_dict[header] = dict()
            for param in pdf_dict[header]:


                param_included = False
                for param_ in re[header]:

                    if param.startswith(param_):
                        detailed_dict[header][param_] = re[header][param_]
                        param_included = True
                        break
                if not param_included:
                    detailed_dict[header][param] = pdf_dict[header][param]

        return detailed_dict

    @staticmethod
    def get_pdf_headers(pdf_headers, pdf_dict):
        for header in pdf_dict.keys():
            if header not in pdf_headers.keys():
                pdf_headers[header] = pdf_dict[header]
            else:
                for param in pdf_dict[header]:
                    pdf_headers[header].append(param)

        return pdf_headers



def get_pickled_pdfs():
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
        try:
            pdf_dict = analyzer.get_pdf_dict(headers, footers, text_tesseract)
            move_files(tds_path, filename, pdf_dict)
        except Exception as e:
            print(e)

def unpickle_dicts():
    analyzer = Analyze_Text()
    pdf_headers = dict()

    filenames, tds_path = get_files_as_array(datatype=".pickle")
    for filename in filenames:
        pdf_dict = pickle_get_content(tds_path + filename)
        pdf_headers = analyzer.get_pdf_headers(pdf_headers, pdf_dict)
        # pdf_dict_detailed = analyzer.get_pdf_dict_detailed(pdf_dict, text_PyMuPDF)

        a=1

    pickle_dump_content(BASE_DIR + '/headers.pickle', pdf_headers)
    # from keyword import Keyword_Analyzer
    # keyword_analyzer = Keyword_Analyzer()
    # headers = keyword_analyzer.analyze(pdf_headers.keys())

    a=1

def main():
    # get_pickled_pdfs()
    unpickle_dicts()


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