
import numpy as np
import pandas as pd
import os
import time
import csv

from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt

# import ssl

# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context

# Import libraries for text preprocessing
import re
import nltk
import seaborn as sns

# You only need to download these resources once. After you run this 
# the first time--or if you know you already have these installed--
# you can comment these two lines out (with a #)
# nltk.download('stopwords')
# nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer



from sklearn.feature_extraction.text import TfidfTransformer 
from sklearn.feature_extraction.text import CountVectorizer


from scipy.sparse import coo_matrix

# https://github.com/andybywire/nlp-text-analysis


from pdf_reader import BASE_DIR






# find keywords in general
class Keyword:
    def __init__(self):
        self.__PATH_CUSTOM_STOP_WORDS = BASE_DIR + '/stop_words.txt'

    def print_status(func):# pylint: disable=no-self-argument
        def wrapper(self,*args,**kwargs):
            func_name = func.__name__# pylint: disable=no-member
            if func_name.startswith('__'):
                func_name = func_name[2:]
            elif func_name.startswith('_'):
                func_name = func_name[1:]

            start_time = time.perf_counter()
            print(F'\tstart {func_name}')

            return_ = func(self,*args,**kwargs)# pylint: disable=not-callable

            end_time = time.perf_counter()
            used_time = int((end_time - start_time) * 100) / 100

            print(F'done {func_name}\t in {used_time}s')

            return return_
        return wrapper


    def analyze(self, input_, list_=None, content_dict=None, print_details=False):
        if True:
            corpus,stop_words,ds_count = self.__get_dataframe(input_)

            self.__get_word_cloud(corpus,stop_words,print_details=print_details)
            X, tokenized_vocabulary, cv = self.__tokenize_vocabulary(corpus,stop_words)

            extractor = Extractor()
            number_of_words = 100
            most_freq_words = extractor.extract_most_freq_words(corpus, n_in=number_of_words)
            most_freq_bigrams = extractor.extract_most_freq_bigrams(corpus, n_in=number_of_words)
            most_freq_trigrams = extractor.extract_most_freq_trigrams(corpus, n_in=number_of_words)

            tf_idf = TF_IDF()
            tf_idf = tf_idf.get_tf_idf(corpus, cv, X, ds_count)

        self.__save_result_in_several_csvs_with_full_details()


        return tokenized_vocabulary,most_freq_words,most_freq_bigrams,most_freq_trigrams,tf_idf

    @print_status
    def __get_dataframe(self, input_, column_names_to_drop=None, content_dict=None, print_details=False):
        if type(input_) == str:
            df = pd.read_csv(input_, delimiter=',')
            # print(df.head)
            df = df.drop(column_names_to_drop, axis=1)
            # print(df.describe)
        elif type(input_) == list:
            df = pd.DataFrame(input_, columns=['data'])
            print(df.head)
            print(df.describe)
        else:
            return



        # View 10 most common words prior to text pre-processing
        freq = pd.Series(' '.join(map(str, df[self.__datacol])).split()).value_counts()[:10]
        # View 10 least common words prior to text pre-processing
        freq1 =  pd.Series(' '.join(map(str,df[self.__datacol])).split()).value_counts()[-10:]

        if print_details: print(freq)
        if print_details: print(freq1)

        #

        corpus,stop_words,ds_count = self.__clean_dataset(df,print_details=print_details)

        return corpus,stop_words,ds_count

    @print_status
    def __clean_dataset(self,df,print_details=False):
        # Create a list of stop words from nltk
        nltk.data.path.append(os.path.join(BASE_DIR, 'data/nltk_data'))
        stop_words = set(stopwords.words("english"))
        if print_details: print(sorted(stop_words))

        # Load a set of custom stop words from a text file (one stopword per line)
        csw = set(line.strip() for line in open(self.__PATH_CUSTOM_STOP_WORDS))
        csw = [sw.lower() for sw in csw]
        if print_details: print(sorted(csw))

        # Combine custom stop words with stop_words list
        stop_words = stop_words.union(csw)
        if print_details: print(sorted(stop_words))




        # Pre-process dataset to get a cleaned and normalised text corpus
        corpus = []
        df['word_count'] = df[self.__datacol].apply(lambda x: len(str(x).split(" ")))
        ds_count = len(df.word_count)
        for i in range(0, ds_count):
            # Remove punctuation
            text = re.sub('[^a-zA-Z]', ' ', str(df[self.__datacol][i]))
            
            # Convert to lowercase
            text = text.lower()
            
            # Remove tags
            text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)
            
            # Remove special characters and digits
            text=re.sub("(\\d|\\W)+"," ",text)
            
            # Convert to list from string
            text = text.split()
            
            # Stemming
            ps=PorterStemmer()
            
            # Lemmatisation
            lem = WordNetLemmatizer()
            text = [lem.lemmatize(word) for word in text if not word in  
                    stop_words] 
            text = " ".join(text)
            corpus.append(text)

        return corpus,stop_words,ds_count

    @print_status
    def __get_word_cloud(self,corpus,stop_words,print_details=False):
        # Generate word cloud
        # %matplotlib inline # only for jupyter notebooks
        wordcloud = WordCloud(
                                background_color='white',
                                stopwords=stop_words,
                                max_words=100,
                                max_font_size=50, 
                                random_state=42
                                ).generate(str(corpus))
        if print_details: print(wordcloud)
        fig = plt.figure(1)
        plt.imshow(wordcloud)
        plt.axis('off')
        # plt.show()
        wordcloud_path = os.path.abspath(os.path.dirname(__file__)) + '/results/wordcloud.png'
        fig.savefig(wordcloud_path, dpi=900)

    @print_status
    def __tokenize_vocabulary(self,corpus,stop_words):
        # Tokenize the text and build a vocabulary of known words

        cv = CountVectorizer(max_df=0.8,stop_words=stop_words, max_features=10000, ngram_range=(1,3))
        X = cv.fit_transform(corpus)


        # Sample the returned vector encoding the length of the entire vocabulary
        tokenized_vocabulary = list(cv.vocabulary_.keys())[:10]
        return X, tokenized_vocabulary, cv


    @print_status
    def __save_result_in_several_csvs_with_full_details(self):
        df_all = pd.read_csv(self.__path, delimiter=',')
        result_list = []#df_all[0:0]#pd.DataFrame()#df_all[0:0]

        directory = os.path.abspath(os.path.dirname(__file__)) + "/results/"
        for file in os.listdir(directory):
            if file.endswith(".csv") and not file.endswith("_.csv"):
                file_name = os.path.join(directory, file)
                df = pd.read_csv(file_name, delimiter=',')
                if df.empty:
                    print(F'{file_name} is empty')
                else:
                    print(F'{file_name} start collecting')
                    for index,keyword in enumerate(df.iloc[:,1]):
                        keywords = '|'.join(keyword.split(' '))
                        df_found = df_all[df_all[self.__datacol].str.contains(keywords, na=False)].copy(deep=True)

                        df_found['keyword'] = str(keyword)
                        amount_column_name = df.columns[2].lower()
                        amount_value = df.iloc[index,2]
                        if type(amount_value) == np.int64:
                            amount_value = int(amount_value)
                        elif type(amount_value) == np.float64:
                            amount_value = float(amount_value)
                        else:
                            print(type(amount_value))

                        df_found[amount_column_name] = amount_value

                        if not df_found.empty:
                            try:
                                df_result = pd.concat([df_result,df_found])
                            except NameError:
                                df_result = df_found.copy(deep=True)
                        else:
                            print(f'\t no rows found for keyword: {keyword}')
                    
                    df_result.to_csv(file_name.replace('.csv','_.csv'))
                    print(f'{len(df_result.index)}\t found rows containing keywords')
                    try:
                        df_overall_result = pd.concat([df_overall_result,df_result])
                    except NameError:
                        df_overall_result = df_result.copy(deep=True)
        
        df_overall_result.to_csv(os.path.abspath(os.path.dirname(__file__)) + '/data/data_result_keywords.csv')
        return df_overall_result

# get mono-, bi- and trigrams
class Extractor:
    def __init__(self):
        pass

    # MONO
    def extract_most_freq_words(self,corpus, n_in=None, print_details=False):
        # Convert most freq words to dataframe for plotting bar plot, save as CSV
        if n_in == None:
            n = 20
        else:
            n = n_in
        top_words = self.__get_top_n_words(corpus, n=n)
        top_df = pd.DataFrame(top_words)
        top_df.columns=["Keyword", "Frequency"]
        if print_details: print(top_df)
        top_df.to_csv(os.path.abspath(os.path.dirname(__file__)) + '/results/top_words.csv')

        # Barplot of most freq words
        sns.set(rc={'figure.figsize':(13,8)})
        g = sns.barplot(x="Keyword", y="Frequency", data=top_df, palette="Blues_d")
        g.set_xticklabels(g.get_xticklabels(), rotation=45)
        g.figure.savefig(os.path.abspath(os.path.dirname(__file__)) +  '/results/keyword.png', bbox_inches = "tight")

        return top_df
        
    @staticmethod
    def __get_top_n_words(corpus, n=None):
        vec = CountVectorizer().fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in      
                    vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], 
                        reverse=True)
        return words_freq[:n]


    # BI
    def extract_most_freq_bigrams(self,corpus, n_in=None, print_details=False):
        # Convert most freq bigrams to dataframe for plotting bar plot, save as CSV
        if n_in == None:
            n = 20
        else:
            n = n_in
        top2_words = self.__get_top_n2_words(corpus, n=n)
        top2_df = pd.DataFrame(top2_words)
        top2_df.columns=["Bi-gram", "Frequency"]
        if print_details: print(top2_df)
        top2_df.to_csv(os.path.abspath(os.path.dirname(__file__)) +  '/results/bigrams.csv')

        # Barplot of most freq Bi-grams
        sns.set(rc={'figure.figsize':(13,8)})
        h=sns.barplot(x="Bi-gram", y="Frequency", data=top2_df, palette="Blues_d")
        h.set_xticklabels(h.get_xticklabels(), rotation=75)
        h.figure.savefig(os.path.abspath(os.path.dirname(__file__)) +  "/results/bi-gram.png", bbox_inches = "tight")

        return top2_df

    @staticmethod
    def __get_top_n2_words(corpus, n=None):
        vec1 = CountVectorizer(ngram_range=(2,2),  
                max_features=2000).fit(corpus)
        bag_of_words = vec1.transform(corpus)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in     
                    vec1.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], 
                    reverse=True)
        return words_freq[:n]


    # TRI
    def extract_most_freq_trigrams(self,corpus, n_in=None, print_details=False):
        # Convert most freq trigrams to dataframe for plotting bar plot, save as CSV
        if n_in == None:
            n = 20
        else:
            n = n_in
        top3_words = self.__get_top_n3_words(corpus, n=n)
        top3_df = pd.DataFrame(top3_words)
        top3_df.columns=["Tri-gram", "Frequency"]
        if print_details: print(top3_df)
        top3_df.to_csv(os.path.abspath(os.path.dirname(__file__)) + '/results/trigrams.csv')

        # Barplot of most freq Tri-grams
        sns.set(rc={'figure.figsize':(13,8)})
        j=sns.barplot(x="Tri-gram", y="Frequency", data=top3_df, palette="Blues_d")
        j.set_xticklabels(j.get_xticklabels(), rotation=75)
        j.figure.savefig(os.path.abspath(os.path.dirname(__file__)) + "/results/tri-gram.png", bbox_inches = "tight")

        return top3_df

    @staticmethod
    def __get_top_n3_words(corpus, n=None):
        vec1 = CountVectorizer(ngram_range=(3,3), 
            max_features=2000).fit(corpus)
        bag_of_words = vec1.transform(corpus)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in     
                    vec1.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], 
                    reverse=True)
        return words_freq[:n]

# Term frequency-inverse document frequency
class TF_IDF:
    def __init__(self):
        pass

    def get_tf_idf(self,corpus,cv,X,ds_count):
        # Get TF-IDF (term frequency/inverse document frequency) -- 
        # TF-IDF lists word frequency scores that highlight words that 
        # are more important to the context rather than those that 
        # appear frequently across documents

        tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
        tfidf_transformer.fit(X)

        # Get feature names
        feature_names=cv.get_feature_names()
        
        # Fetch document for which keywords needs to be extracted
        doc=corpus[ds_count-1]
        
        # Generate tf-idf for the given document
        tf_idf_vector=tfidf_transformer.transform(cv.transform([doc]))

        ##################

        # Sort the tf-idf vectors by descending order of scores
        sorted_items=self.__sort_coo(tf_idf_vector.tocoo())

        # Extract only the top n; n here is 25
        keywords=self.__extract_topn_from_vector(feature_names,sorted_items,25)
        
        # Print the results, save as CSV
        print("\nAbstract:")
        print(doc)
        print("\nKeywords:")
        for k in keywords:
            print(k,keywords[k])

        with open(os.path.abspath(os.path.dirname(__file__)) + '/results/td_idf.csv', 'w', newline="") as csv_file:  
            writer = csv.writer(csv_file)
            writer.writerow(["","Keyword", "Importance"])
            index = 0
            for key, value in keywords.items():
                writer.writerow([index,key, value])
                index += 1

        return keywords

    @staticmethod
    def __sort_coo(coo_matrix):
        # Sort tf_idf in descending order

        tuples = zip(coo_matrix.col, coo_matrix.data)
        return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
    
    @staticmethod
    def __extract_topn_from_vector(feature_names, sorted_items, topn=25):
        
        # Use only topn items from vector
        sorted_items = sorted_items[:topn]
        score_vals = []
        feature_vals = []
        
        # Word index and corresponding tf-idf score
        for idx, score in sorted_items:
            
            # Keep track of feature name and its corresponding score
            score_vals.append(round(score, 3))
            feature_vals.append(feature_names[idx])
    
        # Create tuples of feature,score
        # Results = zip(feature_vals,score_vals)
        results= {}
        for idx in range(len(feature_vals)):
            results[feature_vals[idx]]=score_vals[idx]
        return results

        
    


def main():
    path = os.path.abspath(os.path.dirname(__file__)) + '/data/data_result_s_s.csv'
    path = os.path.abspath(os.path.dirname(__file__)) + '/data/data_result.csv'

    datacol = 'Additional comments'
    keyword = Keyword(path,datacol)
    tokenized_vocabulary,most_freq_words,most_freq_bigrams,most_freq_trigrams,tf_idf = keyword.analyze()

if __name__ == '__main__':
    main()