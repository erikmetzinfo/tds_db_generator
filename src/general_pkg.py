# general_pkg.py
import random
from fuzzywuzzy import fuzz
from fuzzywuzzy import process as fuzzy_process
import Levenshtein
import statistics
import os


def get_dict_key_from_val(dict_, val):
    '''Returns the key of a value from a dictionary.

        Args:
            dict_ (dict): dictionary containing the value
            val (*): value from where the key should be returned

        Returns:
            *: key of the value
    '''
    for key, value in dict_.items():
        if val == value:
            return key

def get_most_frequent_element_of_list(list_):
    '''Returns the most frequent element of a list.

        Args:
            list_ (list): list to search for the most frequent element

        Returns:
            *: most frequent element
    '''
    counter = 0
    most_frequent_element = list_[0]

    for element in list_:
        curr_frequency = list_.count(element)
        if curr_frequency > counter:
            counter = curr_frequency
            most_frequent_element = element

    return most_frequent_element

def unicode_to_ascii(string):
    '''Returns an ascii coded string from an unicoded string.

        Args:
            string (str): unicoded string

        Returns:
            str: ascii coded string
    '''
    ascii_string = (string.
                    replace('\\xe2\\x80\\x99', "'").
                    replace('\\xc3\\xa9', 'e').
                    replace('\\xe2\\x80\\x90', '-').
                    replace('\\xe2\\x80\\x91', '-').
                    replace('\\xe2\\x80\\x92', '-').
                    replace('\\xe2\\x80\\x93', '-').
                    replace('\\xe2\\x80\\x94', '-').
                    replace('\\xe2\\x80\\x94', '-').
                    replace('\\xe2\\x80\\x98', "'").
                    replace('\\xe2\\x80\\x9b', "'").
                    replace('\\xe2\\x80\\x9c', '"').
                    replace('\\xe2\\x80\\x9c', '"').
                    replace('\\xe2\\x80\\x9d', '"').
                    replace('\\xe2\\x80\\x9e', '"').
                    replace('\\xe2\\x80\\x9f', '"').
                    replace('\\xe2\\x80\\xa6', '...').
                    replace('\\xe2\\x80\\xb2', "'").
                    replace('\\xe2\\x80\\xb3', "'").
                    replace('\\xe2\\x80\\xb4', "'").
                    replace('\\xe2\\x80\\xb5', "'").
                    replace('\\xe2\\x80\\xb6', "'").
                    replace('\\xe2\\x80\\xb7', "'").
                    replace('\\xe2\\x81\\xba', "+").
                    replace('\\xe2\\x81\\xbb', "-").
                    replace('\\xe2\\x81\\xbc', "=").
                    replace('\\xe2\\x81\\xbd', "(").
                    replace('\\xe2\\x81\\xbe', ")").
                    replace('\xa0', " ").
                    replace('â\x84¢', "™").
                    replace('ÂÂ®', "®").
                    replace('Â®', "®").
                    replace('ÃÃ©', 'é').
                    replace('Ã©', '©').
                    replace(" 20â\x80\x9d", "").
                    replace(" 20â\\x80\\x9d", "").
                    replace("\xe2\x80\x99", "'").
                    replace("\\xe2\\x80\\x99", "'").
                    replace(" â\\x80\\x93", "").
                    replace(" â\\x80\\x93", "").
                    replace("â\x80\x8b", "").
                    replace("â\x80\x9c", '"').
                    replace("â\x80\x9d", '"').
                    replace("Ç\x80", "ǀ").
                    replace("â\x80\x93", "-").
                    replace("\\xc3\\xa9", "e")
                    )

    return ascii_string

def ascii_to_unicode(string):
    '''Returns an unicoded coded string from an ascii string.

        Args:
            string (str): ascii string

        Returns:
            str: unicoded coded string
    '''
    unicode_string = (string.
                        replace("'", '\\xe2\\x80\\x99').
                        # replace('e', '\\xc3\\xa9').
                        replace('-', '\\xe2\\x80\\x90').
                        replace('-', '\\xe2\\x80\\x91').
                        replace('-', '\\xe2\\x80\\x92').
                        replace('-', '\\xe2\\x80\\x93').
                        replace('-', '\\xe2\\x80\\x94').
                        replace('-', '\\xe2\\x80\\x94').
                        replace("'", '\\xe2\\x80\\x98').
                        replace("'", '\\xe2\\x80\\x9b').
                        replace("'", '\\xe2\\x80\\x9c').
                        replace("'", '\\xe2\\x80\\x9c').
                        replace("'", '\\xe2\\x80\\x9d').
                        replace("'", '\\xe2\\x80\\x9e').
                        replace("'", '\\xe2\\x80\\x9f').
                        replace('...', '\\xe2\\x80\\xa6').
                        replace("'", '\\xe2\\x80\\xb2').
                        replace("'", '\\xe2\\x80\\xb3').
                        replace("'", '\\xe2\\x80\\xb4').
                        replace("'", '\\xe2\\x80\\xb5').
                        replace("'", '\\xe2\\x80\\xb6').
                        replace("'", '\\xe2\\x80\\xb7').
                        replace("+", '\\xe2\\x81\\xba').
                        replace("-", '\\xe2\\x81\\xbb').
                        replace("=", '\\xe2\\x81\\xbc').
                        replace("(", '\\xe2\\x81\\xbd').
                        replace(")", '\\xe2\\x81\\xbe').
                        # replace(" ", "\xa0").
                        replace("™", "â\x84¢").
                        replace("®", "Â®").
                        replace("©", "Ã©")

                        )
    return unicode_string

def string_comparison(string1, string2, max_value=99):
    '''Returns the average equality of two strings calculated
        by the Levenshtein algorithm.

        Args:
            string1 (str): first string to compare with second string
            string2 (str): second string to compare with first string
            max_value (int): maximal permitted comparison average
                (default is 99)

        Returns:
            (bool,float): tuple where the first argument returns the
                equality as bool and the second argument with the
                comparison average as integer
    '''
    # https://www.datacamp.com/community/tutorials/fuzzy-string-python
    average = fuzz.token_set_ratio(string1, string2)

    average_rounded = int(average * 100) / 100
    if average_rounded >= max_value:
        return True, average_rounded
    else:
        return False, average_rounded

def string_list_comparison(string, string_list, max_value=99):
        '''Returns the average equality of a string and a list of strings calculated
            by the Levenshtein algorithm.

            Args:
                string1 (str): first string to compare with string list
                string_list (str): string list to compare with first string
                max_value (int): maximal permitted comparison average
                    (default is 99)

            Returns:
                (bool,float): tuple where the first argument returns the
                    equality as bool and the second argument with the
                    comparison average as integer
        '''
        ratio_list = fuzzy_process.extract(string, string_list)
        ratios = []
        for ratio_tuple in ratio_list:
            ratios.append(ratio_tuple[1])

        average = statistics.mean(ratios)
        average_rounded = int(average * 100) / 100
        if average >= max_value:
            return True, average_rounded
        else:
            return False, average_rounded


