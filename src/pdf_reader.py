import os
import sys

import PyPDF2 as pypdf
# from pdfminer.high_level import extract_text_to_fp
# from pdfminer.layout import LAParams
from typing import BinaryIO
from bs4 import BeautifulSoup
import pandas as pd
import pytesseract as pt
import pdf2image
import fitz

class Pdf_reader(object):
    def __init__(self):
        self.__BASE_DIR = os.path.abspath(os.path.dirname(__file__))

    @staticmethod
    def get_general_data_of_pdf(filepath: str) -> dict:
        info_dict = dict()
        with open(filepath,'rb') as f:
            pdf = pypdf.PdfFileReader(f)
            info = pdf.getDocumentInfo()
            number_of_pages = pdf.getNumPages()

            info_dict['author'] = info.author
            info_dict['creator'] = info.creator
            info_dict['producer'] = info.producer
            info_dict['subject'] = info.subject
            info_dict['title'] = info.title
            info_dict['number_of_pages'] = pdf.getNumPages()

        return info_dict

    @staticmethod
    def save_pdf_as_img_with_PyMuPDF(filepath: str, filename: str, to_filepath: str, img_output_type: str='png', dpi: int=200):
        """
        :param output_type: pam, pgm, pgm, png, pnm, ppm, ps, psd
        """
        doc = fitz.open(filepath + filename)
        for page in doc:
            pixmap = page.get_pixmap(img_output_type, dpi=dpi)
            pixmap.writeImage(f'{to_filepath}{page.number}.png')
    
    @staticmethod
    def save_pdf_as_img_with_pdf2image(filepath: str, filename: str, to_filepath: str, img_output_type: str='png', dpi: int=200):
        """
        :param output_type: ppm, jpeg, png, tiff
        """
        pages = pdf2image.convert_from_path(pdf_path=filepath + filename, dpi=dpi, fmt=img_output_type)
        for i in range(len(pages)):
            pages[i].save(f'{to_filepath}{i}.jpg')

    def get_text_from_pdf(self, filepath: str, filename: str, method: int=1, output_type: str='text'):
        """
        :param method: 1=pdf_miner, 2=PyMuPDF, 3=PyPDF2, 4=tesseract
        :param output_type: text, html, xml
        """
        if method == 1:
            return self.__extract_text_from_pdf_with_pdf_miner(filepath + filename,output_type)
        elif method == 2:
            return self.__extract_text_from_pdf_with_PyMuPDF(filepath, filename, output_type)
        elif method == 3:
            return self.__extract_text_from_pdf_with_PyPDF2(filepath, filename)
        elif method == 4:
            return self.__extract_text_from_pdf_with_tesseract(filepath, filename)
        else:
            return ''

    @staticmethod
    def __extract_text_from_pdf_with_pdf_miner(pdf_fo: BinaryIO, output_type:str) -> str:
        """
        https://pdfminersix.readthedocs.io/_/downloads/en/latest/pdf/
        Extracts text from a PDF

        :param pdf_fo: a byte file object representing a PDF file
        :param output_type: text, html, xml, tag
        :return: extracted text
        :raises pdfminer.pdftypes.PDFException: on invalid PDF
        """
        from io import StringIO
        import pdfminer
        out_fo = StringIO()
        pdfminer.high_level.extract_text_to_fp(pdf_fo, out_fo, laparams=pdfminer.layout.LAParams(), output_type=output_type, codec=None)
        return out_fo.getvalue()

    @staticmethod
    def __extract_text_from_pdf_with_PyMuPDF(filepath: str, filename: str, output_type:str) -> list:
        """
        :param output_type: text, blocks, words, html, dict, json, rawdict, rawjson, xhtml, xml
        """
        text_list = list()
        doc = fitz.open(filepath + filename)
        for page in doc:
            text_list.append(page.get_text(output_type))

        return text_list

    @staticmethod
    def __extract_text_from_pdf_with_PyPDF2(filepath: str, filename: str) -> list:
        text_list = set()
        with open(filepath,'rb') as f:
            pdf = pypdf.PdfFileReader(f)

            for page in pdf.pages:
                text = page.extractText()
                text_list.add(text)

        return text_list

    @staticmethod
    def __extract_text_from_pdf_with_tesseract(filepath: str, filename: str, lang: str='eng', dpi: int=200) -> list:
        text_list = set()
        pages = pdf2image.convert_from_path(pdf_path=filepath + filename, dpi=dpi)
        for i in range(len(pages)):
            text_list.add(pt.image_to_string(pages[i], lang=lang))

        return text_list

"""
class Pdf_analyzer(object):
    def __init__(self):
        self.__EXCLUDE_CHARACTERS = {'',' ','\n'}    

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

    def get_values(div_container):
        span_parameters = div_container.find_all('span', {'style':'font-family: BookmanOldStyle-Bold; font-size:10px'})
        if span_parameters != []:
            a = 1
        for span_parameter in span_parameters:
            parameter = span_parameter.text
            parameter = parameter.replace('\n','').strip()
            if parameter not in EXCLUDE_CHARACTERS:
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

    def delete_no_info_rows(rows):
        index = 0
        for i in range(len(rows)):
            value = rows[index].strip()
            if value in EXCLUDE_CHARACTERS:
                rows.pop(index)
            else:
                rows[index] = value
                index += 1

        return rows

    def get_header_rows(rows, headers):
        header_rows = dict()
        for index, row in enumerate(rows):
            #if row in headers:
            if any(map(row.__contains__, headers)):
                header_rows[index] = row

        return header_rows

    def get_result_dict(header_rows, rows, headers):
        result_dict = dict()
        for index, key in enumerate(header_rows.keys()):
            header = header_rows[key]
            header_row = key

            if index == len(header_rows.keys()) - 1:
                #break
                next_header_row = len(header_rows.keys())
            else:
                next_header_row = list(header_rows.keys())[index + 1]
            for row in range(header_row + 1, next_header_row):
                value = rows[row]
                if header not in result_dict.keys():
                    result_dict[header] = list()
                result_dict[header].append(value)

        return result_dict

"""


'''

pdf2txt.py /Users/alpha/Documents/GitHub/tds_db_generator/src/data/tds/Autofroth 1402-1 Resin & Autofroth 10000A Isocyanate.pdf

'''