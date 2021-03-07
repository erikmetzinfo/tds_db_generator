import os
import sys

import pandas as pd

from pdf_reader import Pdf_reader


# Files
BASE_DIR = os.path.abspath(os.path.dirname(__file__) )
def get_files_as_array(datatype: str='pdf') -> list:
    TDS_PATH = BASE_DIR + '/data/tds_pdf/'
    return [f for f in os.listdir(TDS_PATH) if f.endswith(datatype) if os.path.isfile(os.path.join(TDS_PATH, f))], TDS_PATH


def main():
    pdf_reader = Pdf_reader()
    filenames, tds_path = get_files_as_array(datatype=".pdf")
    for filename in filenames:
        print(tds_path + filename)

        # get general data
        pdf_gerneral_info_dict = pdf_reader.get_general_data_of_pdf(tds_path + filename)

        # get text
        text_pdf_miner = pdf_reader.get_text_from_pdf(tds_path, filename, method=1, outpu_type='text')
        text_PyMuPDF = pdf_reader.get_text_from_pdf(tds_path, filename, method=2, outpu_type='text')
        text_PyPDF2 = pdf_reader.get_text_from_pdf(tds_path, filename, method=3, outpu_type='text')
        text_tesseract = pdf_reader.get_text_from_pdf(tds_path, filename, method=4, outpu_type='text')
        
        # save pdf as image
        to_filepath = os.path.join(tds_path.replace('tds_pdf','tds_img'), filename.replace(' ','_'))
        img_output_type = 'png'
        dpi = 200
        pdf_reader.save_pdf_as_img_with_PyMuPDF(tds_path, filename, to_filepath, img_output_type=img_output_type, dpi=dpi)
        pdf_reader.save_pdf_as_img_with_pdf2image(tds_path, filename, to_filepath, img_output_type=img_output_type, dpi=dpi)
        a = 999

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