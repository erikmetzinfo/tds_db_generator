
from general_pkg import string_comparison
from fuzzywuzzy import process as fuzzy_process

a = 'Ratio by weight ( A / B ) 100/ 9505'
a = 'Ratio by weight ( A / B ) 100/ 9505'
b = 'Ratio by weight ( A / B )'

match, match_ratio = string_comparison(a,b,max_value=95)

def special_string_comparison(string1, string2):
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

val = special_string_comparison(a,b)
x=1