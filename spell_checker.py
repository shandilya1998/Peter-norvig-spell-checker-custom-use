# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 15:04:12 2019
Spell check testing script
generates a given number of test questions based on a 6 six templates
tests them on the  modified spell checker and
returns average accuracy over all test sentences
@author: shreyas.shandilya
"""
import random
import numpy as np
import re
import copy
import pandas as pd
import pickle
import time
def damerau_levenshtein_distance(seq1, seq2):
    """Calculate the Damerau-Levenshtein distance between sequences.
    This method has not been modified from the original.
    Source: http://mwh.geek.nz/2009/04/26/python-damerau-levenshtein-distance/
    This distance is the number of additions, deletions, substitutions,
    and transpositions needed to transform the first sequence into the
    second. Although generally used with strings, any sequences of
    comparable objects will work.
    Transpositions are exchanges of *consecutive* characters; all other
    operations are self-explanatory.
    This implementation is O(N*M) time and O(M) space, for N and M the
    lengths of the two sequences.
    >>> dameraulevenshtein('ba', 'abc')
    2
    >>> dameraulevenshtein('fee', 'deed')
    2
    It works with arbitrary sequences too:
    >>> dameraulevenshtein('abcd', ['b', 'a', 'c', 'd', 'e'])
    2
    """
    # codesnippet:D0DE4716-B6E6-4161-9219-2903BF8F547F
    # Conceptually, this is based on a len(seq1) + 1 * len(seq2) + 1 matrix.
    # However, only the current and two previous rows are needed at once,
    # so we only store those.
    oneago = None
    thisrow = list(range(1, len(seq2) + 1)) + [0]
    for x in range(len(seq1)):
        # Python lists wrap around for negative indices, so put the
        # leftmost column at the *end* of the list. This matches with
        # the zero-indexed strings and saves extra calculation.
        twoago, oneago, thisrow = (oneago, thisrow, [0] * len(seq2) + [x + 1])
        for y in range(len(seq2)):
            delcost = oneago[y] + 1
            addcost = thisrow[y - 1] + 1
            subcost = oneago[y - 1] + (seq1[x] != seq2[y])
            thisrow[y] = min(delcost, addcost, subcost)
            # This block deals with transpositions
            if (x > 0 and y > 0 and seq1[x] == seq2[y - 1]
                    and seq1[x - 1] == seq2[y] and seq1[x] != seq2[y]):
                thisrow[y] = min(thisrow[y], twoago[y - 2] + 1)
    return thisrow[len(seq2) - 1]

def damerau_levenshtein_sim(seq1, seq2):
    len_1=len(seq1)
    len_2=len(seq2)
    sim=(len_1+len_2-damerau_levenshtein_distance(seq1, seq2))/(len_1+len_2)
    return sim

from difflib import SequenceMatcher
class spell_checker():
    def __init__(self,create_dict=False):
        self.file_db=r'C:\Users\shreyas.shandilya\Desktop\KPI_db_updated.csv'
        self.file_dict_wrds_pkl=r'C:\Users\shreyas.shandilya\Desktop\dictionary_words.pickle'
        self.data=pd.read_csv(self.file_db)
        self.create_dictionary()
               
    def create_dictionary(self):
        #Create a dictionary of words from the training corpus and words used in the database
        self.dictionary_db=pd.Series(list(self.data.columns)[1:8])
        self.dictionary_db=self.dictionary_db.append(self.data['Brand'][:],ignore_index=True)
        self.dictionary_db=self.dictionary_db.append(self.data['Country'][:],ignore_index=True)
        self.dictionary_db=self.dictionary_db.apply(str.lower)
        self.words = self._words_(open('big.txt').read())
        self.dictionary_en = [word for word in self.words if word not in set(self.dictionary_db)]
        self.dictionary=self.dictionary_db.append(pd.Series(self.dictionary_en),ignore_index=True)
        self.dictionary=list(set(self.dictionary))
        
    def rec_dictionary(self):
        px=open(self.file_dict_wrds_pkl,'rb')
        self.dictionary=list(pickle.load(px))
        px.close()
        
    def _words_(self,text): 
        #returns a  list of words from  a given string
        return re.findall(r'\w+', text.lower())
    
    def edits_dist_1(self,word):
        #returns a set of all possible variations of the word obtained after 1 edits
        letters    = 'abcdefghijklmnopqrstuvwxyz'
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)
    
    def edits_dist_2(self,word): 
#        #returns a set of all possible variations of the word obtained after 2 edits
        return set([e2 for e1 in self.edits_dist_1(word) for e2 in self.edits_dist_1(e1)])
        
    def vowel_swap(self,word):
        #returns a list of the letters in the word with vowels replaced by a list vowels
        vowels='aeiouy'
        word=list(word)
        for idx,l in enumerate(word):
            if l in vowels:
                word[idx]='#'
        for idx,l in enumerate(word):
            if l=='#':
                word[idx]=list(vowels)
        return word
    
    def reduce_repeats(self,word):
        #returns a list of the letters in the word with repeated letters replaced by reductions of the repeated letters, if there are repeats, else returns None
        word=list(word)
        idx=0
        word_=[]
        lngth=len(word)
        idy=0
        while(idx<lngth):
            #check if a letter is being repeated
            no_occrnc=1
            while(True):
                try:
                    if(word[idx+no_occrnc]==word[idx] ):
                        no_occrnc=no_occrnc+1
                    else:
                        break
                except:
                    break
            if no_occrnc>1:
                lst_repeats=[word[idx]*i for i in range(1,no_occrnc+1)]
                word_=word_[:idy]+[lst_repeats]
                idx=no_occrnc+idx
                idy=idy+1
            else:
                word_=word_+[word[idx]]
                idx=idx+1
                idy=idy+1
        return word_

    def create_variants(self,word,variant_type):
        #returns a set of all possible variants of the word according to the rules in the methods edit_dist_1(),edit_dist_2(),edit_dist_3(), vowel_swap() and reduce_repeats()
        start=time.time()
        ed1=self.edits_dist_1(word)
        ed2=self.edits_dist_2(word)
        word_lst_vwl=self.create_set(self.vowel_swap(word))
        word_lst_red=self.create_set(self.reduce_repeats(word))
        print('time taken to create variants :')
        print(time.time()-start)
        if variant_type==1:
            return ed1
        elif variant_type==4:
            return ed2
        elif variant_type==3:
            return word_lst_vwl
        elif variant_type==2:
            return word_lst_red
        
    def create_set(self,wrd_lst):
        #returns set of all possible variants created by swapping all all vowels or taking cases with all possible letter repitions
        temp_word=[]
        lst_pos={}
        _lst_=[]
        for idx,l in enumerate(wrd_lst):
            if type(l)==list:
                temp_word.append('#')
                lst_pos[idx]=l
            else:
                temp_word.append(l)
        def fill():
            def subs(words,lst_char,key):
                lst_=[]
                for ch in lst_char:
                    for word in words:
                        word[key]=ch
                        lst_.append(copy.deepcopy(word))
            
                return lst_
            words=[copy.deepcopy(temp_word)]
            for key in lst_pos.keys():
                words=subs(words,lst_pos[key],key)
            return words
                    
        def create_strings(wrd_lst):
            #convert  words in  list form to strings
            lst=[]
            for i in wrd_lst:
                i=[str(j) for j in i]
                temp=''.join(i)
                lst.append(temp)
            return lst 
        _lst_=fill()
        return set(create_strings(_lst_))         
    
    def known_variants(self,word):
        variants=list(self.create_variants(word,1))+[word]
        tmp = [word_ for word_ in variants if word_.lower() in self.dictionary]
        if tmp:
            return tmp
        variants=list(self.create_variants(word,2))
        tmp = [word_ for word_ in variants if word_.lower() in self.dictionary]
        if tmp:
            return tmp
        variants=list(self.create_variants(word,3))
        tmp = [word_ for word_ in variants if word_.lower() in self.dictionary]
        if tmp:
            return tmp
        variants=list(self.create_variants(word,4))
        tmp = [word_ for word_ in variants if word_.lower() in self.dictionary]
        if tmp:
            return tmp       
        
    def selection_criteria(self,word, variant):
        return SequenceMatcher(None, word, variant).ratio()
    
    def candidates(self,word):
        #returns a tuple of closest match to the word and its score
        print(word)
        start=time.time()
        score=[(variant,self.selection_criteria(word,variant)) for variant in self.known_variants(word)]
        print('time taken to compute scores for possible candidates')
        print(time.time()-start)
        print('Number of possible candidates for correction :')
        print(len(score))
        return score

    def find_max(self,score):
        #takes a tuple of variants and their respective score and returns the tuple with the maximum score
        max=0
        idx=0
        for i in range(len(score)):
            if score[i][1]>=max:
                max=score[i][1]
                idx=i
        return(score[idx])
        
    def spell_check(self,sent):
        sent=sent.split(' ')
        abb=re.compile(r'[[a-zA-z]\.]+')
        correction_=[]
        correction=''
        for word_ in sent:
            if word_.isdigit():#check for numbers
                correction_.append(word_)
            elif bool(abb.match(word_)) :#check for the presence of abbreviation
                correction_.apend(word_)
            else:
                candidates=self.candidates(word_.lower())
                print(candidates)
                candidate_df=pd.DataFrame(candidates,columns=['candidate','score'])
                flag_db_wrds=[1 if candidate[0] in self.dictionary_db.values else 0 for candidate in candidates ]
                candidate_df.loc[:,'flag_db_words']=pd.Series(flag_db_wrds)
                flag=candidate_df['flag_db_words'].sum() 
                if flag==0:
                    candidate_df=candidate_df.sort_values(by='score',ascending=False)
                    correction=candidate_df.iloc[0]['candidate']
                else:
                    candidate_df['score_']=candidate_df['score']+candidate_df['flag_db_words']
                    candidate_df=candidate_df.sort_values(by='score_')
                    correction=candidate_df.iloc[0]['candidate']
                correction_.append(correction)
                #return for case when source code of spellchecker is changed
        return ' '.join(correction_)
    
class spell_check_test():
    def __init__(self,no_ques):#input the number of test questions to test the spell checker against
        """
        # - Brand name
        $ - Country
        % - Quarter
        @ - Year
        & - Information required one of the following - [MROI, Sales, Profit,'ACCT','ACR','ACV','AGI','AGR','A/R','BS','BGT','COGS','CPTAL','EPS','FIFO','GL','GP','LC','LIFO','MTD','NAV','OC','PC','Pd','P/E','P&L', 'RE','ROA','ROE','ROI','WC' ]
        this class creates questions to be answered by the converser by populating the  following templates-
        1. What is the & of # in % in @ in $
        2. Please tell me the & of # in $ in % in @
        3. Share the & of # in $ in % in @
        4. Get the & of # in $ in % in @
        5. Give me the & of # in $ in % in @
        6. Provide me the & of # in $ in % in @
        """
        self.file_spell_test_questions=r'C:\Users\shreyas.shandilya\Desktop\Spellcheck_test.csv'
        self.file_spell_test_wrds_df=r'C:\Users\shreyas.shandilya\Desktop\Spell_checker_test_words.pickle'
        self.file_db=r'C:\Users\shreyas.shandilya\Desktop\KPI_db_updated.csv'
        self.file_spell_test_wrds_df=r'C:\Users\shreyas.shandilya\Desktop\Spell_checker_test_words.pickle'
        self.template=['What is the & of # in % in @ in $','Please tell me the & of # in $ in % in @','Share the & of # in $ in % in @','Get the & of # in $ in % in @','Give me the & of # in $ in % in @','Provide me the & of # in $ in % in @']
        self.no_ques=int(no_ques/len(self.template))
        self.ob_spell_checker=spell_checker()
        self.brand=list(self.ob_spell_checker.data['Brand'].unique())
        self.country=list(self.ob_spell_checker.data['Country'].unique())
        self.information=['MROI', 'profit', 'sales','ROI','OC','ACV']#+['ACCT','ACR','ACV','AGI','AGR','A/R','BS','BGT','COGS','CPTAL','EPS','FIFO','GL','GP','LC','LIFO','MTD','NAV','OC','PC','Pd','P/E','P&L', 'RE','ROA','ROE','ROI','WC' ]
        self.quarter=['1', '2', '3','4']
        self.year=list(self.ob_spell_checker.data['Year'].unique())
        self.test_questions=self.create_questions()
        px=open(self.file_spell_test_wrds_df,'rb')
        self.test_words=pickle.load(px)
        self.test_words=self.test_words.rename(columns={'incorrect_spelling':'misspells','correct_spelling':'correct_sell'})
        px.close()
        
    def create_questions(self):
            #creates a list of valid questions with correct spelling
        questions=map(self.substitute,self.template)
        test_set=[]
        for question in questions:
            temp=[ques for ques in self.create_set(question)]
            test_set.extend(temp)
        return test_set
            
#        questions=map(self.create_set,list(questions))
#        questions = [item for sublist in list(questions) for item in sublist]
        return questions
            
    def substitute(self,sent):
        #Replaces the marker for each of the part of the question with a list of all possible values and returns a list contaning each word of the list and the  list of respective values
        sent=sent.split(' ')
        pos_brand=sent.index('#')
        pos_country=sent.index('$')
        pos_quarter=sent.index('%')
        pos_year=sent.index('@')
        pos_info=sent.index('&')
        if pos_brand != -1:
            sent[pos_brand]=[(ele,brand) for brand in self.brand for ele in list(self.ob_spell_checker.create_variants(brand))]
        if pos_country != -1:
            sent[pos_country]=[(ele,country) for country in self.country for ele in list(self.ob_spell_checker.create_variants(country))]
        if pos_info != -1:
            sent[pos_info]=[(ele,info) for info in self.information for ele in list(self.ob_spell_checker.create_variants(info))]
        if pos_year != -1:
            sent[pos_year]=[(str(year),str(year)) for year in self.year]
        if pos_quarter != -1:
            temp=sent[:pos_quarter]+[[(ele,'quarter') for ele in list(self.ob_spell_checker.create_variants('quarter'))]]+[[(quart,quart) for quart in self.quarter]]+sent[pos_quarter+1:]
            sent=temp
        for idx,l in enumerate(sent):
            if type(l)==list:
                continue
            else:
                sent[idx]=[(ele,l) for ele in list(self.ob_spell_checker.create_variants(l))]
        return sent
    
    def create_set(self,wrd_lst):
        #returns set of all possible variants created by swapping all all vowels or taking cases with all possible letter repitions
        misspell=0
        ctr=0
        while(ctr<self.no_ques):
            temp_sent=[]
            for i in range(len(wrd_lst)):
                misspell=random.randint(0,len(wrd_lst[i])-1)
                temp_sent.append(wrd_lst[i][misspell])
            ctr+=1
            yield temp_sent
        return 
    
    def test_words(self,db_test=False):
        #set db_test to True to test the spell checker on database strings
        if db_test:
            misspell_=[]
            correct_spell=[]
            for  word in self.ob_spell_checker.dictionary :
                temp=list(self.ob_spell_checker.create_variants(word))
                temp_=pd.Series(index=np.arange(len(temp)))
                temp_[:]=word
                misspell_.extend(temp)
                correct_spell.extend(list(temp_))
            misspells=pd.DataFrame({'misspell':misspell_,'correct_spell':correct_spell},columns=['misspell','correct_spell'])
            self.test_words=misspells
        i=0
        corr=0
        j=0
        start=random.randint(0,int(len(self.test_words)/1000))
        while(True):
            try:
                print('word to test:')
                print(misspells['misspell'][start+j])
                correction=spell_checker().spell_check(self.test_words['misspell'][start+j])
                print('correction:')
                print(correction)
                print('Correct spelling:')
                print(misspells['correct_spell'][start+j])
                if correction[0]==misspells['correct_spell'][start+j]:
                    corr+=1
                i+=1
                j+=int(len(self.test_words)/1000)
            except:
                break
        print('accuracy:')
        print((corr/i)*100)
        print('words tested:')
        print(i)
    
    def test_sents(self):
        #Tests the spell checker on the dataframe of test sentences created or on the dataframe of test sentences input by the user
        ob=spell_checker()
        temp=0
        ques_acc=0.0
        corr=0
        tot=0
        for ques in self.test_questions:
            lngth=len(ques)
            i=0
            incorrect_ques=[]
            correction_=[]
            correct=[]
            for word in ques:
                correction=ob.spell_check(word[0])
                incorrect_ques.append(word[0])
                correction_.append(correction)
                correct.append(word[1])
                if correction==word[1].lower():
                    i+=1
                    corr+=1
                tot+=1
            ques_acc+=float(i/lngth)
            print('Incorrect question:')
            print(' '.join(incorrect_ques))
            print('Correction:')
            print(' '.join(correction_))
            print('Correct question:')
            print(' '.join(correct))
            temp+=1
        acc=ques_acc/temp
        print('Average accuracy of spell checker on a sentence:')
        print(acc)
        print('word accuracy:')
        print(float(corr/tot))
        
    def score(self,corrected,correct):
        #compares the corrected spelling output to correct spelling and returns the accuracy
        check=(corrected==correct)
        check=check.apply(self.bool2int)
        no_correct=sum(check)
        if correct: 
            accuracy=no_correct/len(correct)
        else:
            accuracy=no_correct/len(correct)
        return accuracy     
        
    def bool2int(self,ele):
        if ele==True:
            ele=1
        else:
            ele=0
        return ele

