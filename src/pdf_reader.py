import os
import sys

import PyPDF2 as pypdf
# from pdfminer.high_level import extract_text_to_fp
# from pdfminer.layout import LAParams
from typing import BinaryIO
from bs4 import BeautifulSoup
import pandas as pd
import pytesseract as pt
try:
    from PIL import Image
except ImportError:
    import Image
import pdf2image
import fitz

class Pdf_reader(object):
    def __init__(self):
        self.__BASE_DIR = os.path.abspath(os.path.dirname(__file__))
        self.__EXCLUDE_CHARACTERS = {'',' ','\n'}   

    @staticmethod
    def get_general_data_of_pdf_with_pypdf(filepath: str) -> dict:
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
    def get_general_data_of_pdf_with_PyMuPDF(filepath: str) -> dict:
        doc = fitz.open(filepath)
        info_dict = doc.metadata
        
        return info_dict

    @staticmethod
    def save_pdf_as_img_with_PyMuPDF(filepath: str, filename: str, to_filepath: str, img_output_type: str='png', dpi: int=200):
        """
        :param output_type: pam, pgm, pgm, png, pnm, ppm, ps, psd
        """
        doc = fitz.open(filepath + filename)
        for page in doc:
            pixmap = page.get_pixmap(dpi=dpi)
            pixmap.writeImage(f'{to_filepath}{page.number}.{img_output_type}')
    
    @staticmethod
    def save_pdf_as_img_with_pdf2image(filepath: str, filename: str, to_filepath: str, img_output_type: str='png', dpi: int=200):
        """
        :param output_type: ppm, jpeg, png, tiff
        """
        pages = pdf2image.convert_from_path(pdf_path=filepath + filename, dpi=dpi, fmt=img_output_type)
        for i in range(len(pages)):
            pages[i].save(f'{to_filepath}{i}.jpg')

    def clean_data(self, pages_set: set) -> list:
        text_list = list()
        for index, page in enumerate(pages_set):
            page_list = page.split('\n')
            text_list.append(list())
            for line in page_list:
                if line.strip() not in self.__EXCLUDE_CHARACTERS:
                    text_list[index].append(line.strip())

        return text_list


    def get_text_from_pdf(self, filepath: str, filename: str, method: int=1, output_type: str='text', dpi: int=200):
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
            return self.__extract_text_from_pdf_with_tesseract(filepath, filename, dpi=dpi)
        else:
            return ''

    @staticmethod
    def extract_html_from_pdf_with_pdf_miner(pdf_fo: BinaryIO) -> str:
        """
        Extracts text from a PDF

        :param pdf_fo: a byte file object representing a PDF file
        :return: extracted text
        :raises pdfminer.pdftypes.PDFException: on invalid PDF
        """
        from pdfminer.high_level import extraextract_text_to_fpct_pages
        from pdfminer.layout import LAParams
        out_fo = StringIO()
        extract_text_to_fp(pdf_fo, out_fo, laparams=LAParams(), output_type='html', codec=None)
        return out_fo.getvalue()

    def __extract_text_from_pdf_with_pdf_miner(self,filepath: str, output_type:str) -> str:
        """
        https://pdfminersix.readthedocs.io/_/downloads/en/latest/pdf/
        Extracts text from a PDF

        :param pdf_fo: a byte file object representing a PDF file
        :param output_type: text, html, xml, tag
        :return: extracted text
        :raises pdfminer.pdftypes.PDFException: on invalid PDF
        """
        # from pdfminer.high_level import extract_pages
        # for page_layout in extract_pages(filepath):
        #     for element in page_layout:
        #         print(element)

        # from pdfminer.layout import LTTextContainer
        # for page_layout in extract_pages(filepath):
        #     for element in page_layout:
        #         if isinstance(element, LTTextContainer):
        #             print(element.get_text())

        # from pdfminer.layout import LTTextContainer, LTChar
        # for page_layout in extract_pages(filepath):
        #     for element in page_layout:
        #         if isinstance(element, LTTextContainer):
        #             for text_line in element:
        #                 for character in text_line:
        #                     if isinstance(character, LTChar):
        #                         print(character.fontname)
        #                         print(character.size)
        pages_set = set()
        page_text = ''
        from pdfminer.high_level import extract_pages
        from pdfminer.layout import LTTextContainer, LTChar, LTTextLineHorizontal, LTTextLine
        for page_layout in extract_pages(filepath):
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    for text_line in element:
                        font_details = dict()
                        font_details['fontname'] = list()
                        font_details['size'] = list()
                        for character in text_line:
                            if isinstance(character, LTChar):
                                font_details['fontname'].append(character.fontname)
                                font_details['size'].append(character.size)
                        font_line_details = self.__get_details_of_dict(font_details)
                        most_common_fontname_in_line = list(font_line_details['fontname'].keys())[0]
                        most_common_size_in_line = list(font_line_details['size'].keys())[0]
                        if isinstance(text_line, LTTextLine):
                            page_text += text_line.get_text()
            pages_set.add(page_text)

        return pages_set

    @staticmethod
    def __extract_text_from_pdf_with_PyMuPDF(filepath: str, filename: str, output_type:str) -> list:
        """
        :param output_type: text, blocks, words, html, dict, json, rawdict, rawjson, xhtml, xml
        """
        pages_set = set()
        doc = fitz.open(filepath + filename)
        for page in doc:
            pages_set.add(page.get_text(output_type))

        return pages_set

    @staticmethod
    def __extract_text_from_pdf_with_PyPDF2(filepath: str, filename: str) -> list:
        pages_set = set()
        with open(filepath + filename,'rb') as f:
            pdf = pypdf.PdfFileReader(f)

            for page in pdf.pages:
                text = page.extractText()
                pages_set.add(text)

        return pages_set

    def __extract_text_from_pdf_with_tesseract(self, filepath: str, filename: str, lang: str='eng', dpi: int=200, psm: int=6) -> list:
        """
        psm
        Page segmentation modes:
        ok          0    Orientation and script detection (OSD) only.
        not working 1    Automatic page segmentation with OSD.
        not working 2    Automatic page segmentation, but no OSD, or OCR.
        ok          3    Fully automatic page segmentation, but no OSD. (Default)
        good        4    Assume a single column of text of variable sizes.
        weird       5    Assume a single uniform block of vertically aligned text.
        !perfect    6    Assume a single uniform block of text.
        weird       7    Treat the image as a single text line.
        weird       8    Treat the image as a single word.
        weird       9    Treat the image as a single word in a circle.
        weird       10    Treat the image as a single character.
        good        11    Sparse text. Find as much text as possible in no particular order.
        good        12    Sparse text with OSD.
        weird       13    Raw line. Treat the image as a single text line, bypassing hacks that are Tesseract-specific.
        """
        pages_set = set()
        # for i in range(len(pages)):
        #     pages_set.add(pt.image_to_string(pages[i], lang=lang))

        doc = fitz.open(filepath + filename)
        zoom_x = 2.0
        zoom_y = 2.0
        mat = fitz.Matrix(zoom_x, zoom_y)
        for index, page in enumerate(doc):
            pixmap = page.get_pixmap(matrix=mat, colorspace='GRAY', alpha = False)
            image_filepath = f'{self.__BASE_DIR}/temp.png'
            pixmap.writeImage(image_filepath)

            # custom_config = r'--oem 3 --psm 6 outputbase digits'
            custom_config = fr'--psm {psm}'
            text = pt.image_to_string(Image.open(image_filepath), lang=lang, config=custom_config, nice=0)#, output_type='text')
            pages_set.add(text)
            os.remove(image_filepath)

        return pages_set



    @staticmethod
    def __get_details_of_dict(dict_: dict) -> dict:
        re_dict = dict()
        for key in dict_.keys():
            re_dict[key] = dict()
            for attr in dict_[key]:
                if attr not in re_dict[key].keys():
                    re_dict[key][attr] = 1
                else:
                    re_dict[key][attr] += 1

        return re_dict

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