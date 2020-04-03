#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 00:33:15 2020

@author: Qingyang Luo
"""
class triple():
    sub = ""
    pred = ""
    obj = ""
    def __init__(self, Sub, Pred, Obj):
        self.sub = Sub
        self.pred = Pred
        self.obj = Obj
        
    def __str__(self):
        return self.sub + ", " + self.pred + ", " + self.obj + "\n";
    
'''
COMPOUND_KEY = "compound"
AMOND_KEY = "amod"

def str_trim(string):
    return string[1: len(string) - 1]
    
def find_compound(strs, sents):
    sentence_number = int(str_trim(strs[1])) - 1
    sub = str_trim(strs[0]) + " " + str_trim(str_trim(sents[sentence_number]).split(", ")[0])
    return sub

def find_punct(strs, pre_strs, post_strs) :
    if str_trim(pre_strs[2]) == AMOND_KEY:
        post_strs[0] = "'" + str_trim(pre_strs[0]) + "-" + str_trim(post_strs[0]) + "'"
        return post_strs[0]
        '''
        
PUNCTUATION = [",", "-", "."]
CONJ_PUNCTIATION = ["-", "(", ")"]
def process_compound(token):
    '''
    compound = ""
    for child in token.children:
        if str(child) != "-" and not child.is_stop and not child.pos_ == "VERB":
            compound += process_compound(child) + " "
    #if token.dep_ == "nsubj":
    #    return token.text + " " + compound
    if token.dep_ == "conj" or token.dep_ == "pobj" or token.dep_ == "nsubj" or token.dep_ == "nsubjpass":
        return compound
    elif token.dep_ == "compound":
        return compound + token.text + " " + token.head.text
    elif token.dep_ == "dobj":
        return compound + token.text
    else:
        return token.text
        '''
    compound = ""
    for child in token.children:
        if not child.is_stop and child.dep_ == "compound" or child.dep_ == "amod" or child.dep_ == "nmod" or child.text in CONJ_PUNCTIATION:
            compound += process_compound(child) + " "
    compound += token.text
    return compound

def process_case(token):
    sub = token.head
    return triple(sub.text, token.text, sub.head.text)

def process_attribute(token) :
    res = ""
    for child in token.children:
        if(child.is_stop):
            continue
        res += child.text + " "
    return res + token.text
      

def process_verb(token):
    objs = []
    for child in token.children:
        if child.text not in PUNCTUATION and child.pos_ != "ADV":
            objs.append(child)
    '''
    if len(objs) != 2:
        return None
    for i in range(0, 2):
        if objs[i].dep_ == "compound" or objs[i].dep_ == "nsubj":
            objs[i] = process_compound(objs[i])
        elif objs[i].dep_ == "attr":
            objs[i] = process_attribute(objs[i])
        elif objs[i].pos_ != "NOUN":
            return None
    return triple(str(objs[0]), token.text, str(objs[1]))
    '''
    

def process_nsubj(token):
    tripleSet = []
    if token.is_stop:
        return tripleSet
    
    sub = process_compound(token)
    #if token.text not in sub:
    #    sub = token.text + " " + sub
    
    pred = token.head.text
    
    for child in token.head.children:
        
        if child.dep_ == "advmod" and child.pos_ == "ADJ":
            pred += " " + child.text
            for grandchild in child.children:
                if grandchild.dep_ == "prep":
                    for grandgrandchild in grandchild.children:
                        if grandgrandchild.dep_ == "pobj":
                            obj = process_compound(grandgrandchild)
                            if obj == "":
                                obj = grandgrandchild.text
                            tripleSet.append(triple(sub, pred, obj))
                                
        elif child.dep_ == "prep": # like "the performance (of)"
            for grandchild in child.children:
                if grandchild.dep_ == "pobj":
                    obj = find_completed_obj(grandchild)
                    tripleSet.append(triple(sub, pred, obj))
                elif grandchild.dep_ == "pcomp":
                    obj = complete_pcomp(grandchild)
                    tripleSet.append(triple(sub, pred, obj))
                    
        elif child.dep_ == "xcomp":
            if child.pos_ != "NOUN":
                for grandchild in child.children:
                    if grandchild.pos_ == "NOUN":
                        obj = grandchild.text
                        tripleSet.append(triple(sub, pred, obj))
                        
        elif child.dep_ == "dobj":
            obj = process_compound(child)
            tripleSet.append(triple(sub, pred, obj))
            
        elif child.i > token.head.i and child.dep_ == "agent":  # like 'represented (by)" 
            for grandchild in child.children:
                if grandchild.pos_ == "NOUN": 
                #if (grandchild.dep_ == "pobj" or grandchild.dep_ == "dobj" or grandchild.dep_ == "npadvmod"):
                    obj = find_completed_obj(grandchild)
                    tripleSet.append(triple(sub, pred, obj))
                    for conj in grandchild.conjuncts:
                        obj = find_completed_obj(conj)
                        tripleSet.append(triple(sub, pred, obj))
            
        
    return tripleSet

def complete_pcomp(token):
    res = ""
    for child in token.children:
        res += " " + find_completed_obj(child);
    return res


def complete_npadvmod(token):
    res = token.head.text
    
    for child in token.head.children:
        if child.i > token.head.i:
            res += " " + child.text
        elif child.i < token.head.i:
            res = child.text + " " + res

    return res
                   
def find_completed_obj(token):
    
    if token.dep_ == "npadvmod":  #npadvmod???
        return complete_npadvmod(token)
    
    clause = ""
    
    if token.head.dep_ == "pcomp":
        clause += token.head.text + " "
    
    for child in token.children:
        if child.i < token.i and (child.dep_ == "prep" or child.pos_ == "PROPN" or child.pos_ == "NOUN" or child.dep_ == "amod"):
            clause += find_completed_obj(child) + " "
    
    return clause + token.text

def process_noun_clause(token):
    clause = token.text
    for child in token.children:
        if child.i > token.i and (child.dep_ == "prep" or child.pos_ == "PROPN" or child.pos_ == "NOUN" or child.dep_ == "amod"):
            clause += " " + process_noun_clause_except_itself(child)
    return clause


def process_obj(token):
    sub = find_completed_obj(token)
    tripleSet = []
    
    for child in token.children:
        
        if child.i > token.i and (child.dep_ == "prep" or child.dep_ == "acl") and child.pos_ == "VERB" :
            pred = child.text  # like 'including', using, based
            if child.nbor().dep_ == "prep" and child.nbor().head == child: # like "based on"
                child = child.nbor()
                pred += " " + child.text
            for grandchild in child.children:
                if grandchild.dep_ == "pobj" or grandchild.dep_ == "dobj":
                    obj = find_completed_obj(grandchild)
                    #obj = obj[0: -len(grandchild.text)] + " " + process_noun_clause_except_itself(grandchild)
                    tripleSet.append(triple(sub, pred, obj))
                    for conj in grandchild.conjuncts:
                        obj = find_completed_obj(conj)
                        #obj = obj[0: -len(conj.text)] + " " + process_noun_clause_except_itself(conj)
                        tripleSet.append(triple(sub, pred, obj))
        
        elif child.i > token.i and (child.dep_ == "prep" and child.pos_ == "ADP" and child.text == "as"):
            pred = child.text     #likc 'such (as)'
            for grandchild in child.children:
                if grandchild.i < child.i and grandchild.head == child:
                    pred = grandchild.text + " " + pred
                elif grandchild.dep_ == "pobj" or grandchild.dep_ == "dobj":
                    obj = find_completed_obj(grandchild)
                    tripleSet.append(triple(sub, pred, obj))
                    for conj in grandchild.conjuncts:
                        obj = find_completed_obj(conj)
                        tripleSet.append(triple(sub, pred, obj))
            
    return tripleSet            

#import stanza
import spacy
from spacy import displacy
#stanza.download('en')   # This downloads the English models for the neural pipeline
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd
import csv
def testSVOs() :
    
    
    #nlp = stanza.Pipeline() # This sets up a default neural pipeline in English
    #doc = nlp("The principal opposition parties boycotted the polls after accusations of vote rigging, and the only other name on the ballot was a little known challenger from a marginal political party")
    
    nlp = spacy.load("en_core_web_sm")
    
    #=========test=========
    outfile = open("firstdraf.csv","w")
    writer = csv.writer(outfile) 
    writer.writerow(["Abstract_num","Subject","Predicate","Objects"])
    csvfile= open ('./abstract_sents_w_abs_num.csv')
    reader = csv.reader(csvfile)
    data = list(reader)               # Convert to list
    data.pop(0) 
    for row in data:
        abs_num,line =row[0],row[1]
        tripleSet = []       
        doc = nlp(line)

        
        for token in doc:
#             print(token.text, token.dep_, token.pos_, token.head.text, token.head.pos_,
#                     [child for child in token.children], token.shape_, token.is_stop, token.tag_)
            if token.text == "'s":
                tripleSet.append(process_case(token))
            #if token.dep_ == "ROOT":
            '''
            if token.dep_ == "ROOT":
                res = process_verb(token)
                if res != None:
                    tripleSet.append(res)
                    '''
            if token.dep_ == "nsubj" or token.dep_ == "nsubjpass":
                res = process_nsubj(token)
                for t in res:
                    tripleSet.append(t)
                res = process_obj(token)
                for t in res:
                    tripleSet.append(t)
            if token.dep_ == "dobj" or token.dep_ == "pobj" or (token.dep_ == "ROOT" and token.pos_ == "NOUN"):
                for t in process_obj(token):
                    tripleSet.append(t)
                
            
            
        
        '''
        sents = sent.dependencies_string().split("\n")
        for i in range(0, len(sents)) :
            strs = str_trim(sents[i]).split(", ")
            if str_trim(strs[2]) == "punct" and str_trim(strs[0]) == "-":
                pre = str_trim(sents[i-1]).split(", ")
                post = str_trim(sents[i+1]).split(", ")
                post[0] = find_punct(strs, pre, post)
                sents[i+1] = "'" + ', '.join(post) + "'"
                
            print(str_trim(strs[0]))
            if str_trim(strs[0]) in entities:
                sub = str_trim(strs[0])
                if str_trim(strs[2]) == 'nmod:poss':
                    number_obj = int(str_trim(strs[2]))
                    obj = str_trim(str_trim(sents[number_obj]).split(", ")[0])
                if str_trim(str_trim(sents[i+1]).split(", ")[2]) == 'case':
                    pred = str_trim(str_trim(sents[i+1]).split(", ")[0])
                tripleSet.append(triple(sub, obj, pred))
                    
            
            if str_trim(strs[2]) == COMPOUND_KEY:
                sub = find_compound(strs, sents)
                print (sub)
                '''
                
                
        #tripleSet.append(triple("g", "'s", "h"))
        
        # Print triples
        #print("--------------------")
        
        for t in range(0, len(tripleSet)):
            #print(tripleSet[t])
            subject,predicte,objects = str(tripleSet[t]).split(",")
            writer.writerow([abs_num,subject,predicte,objects])
          
        
        
    outfile.close()
    
#with preposition
# def process():
#     df =pd.read_csv('./firstdraf.csv')
#     predicates = df['Predicate']
#     wnl = WordNetLemmatizer()
#     after_preprocess =[]
#     for p in predicates:
#         after_preprocess.append(wnl.lemmatize(p),'v')
#     df.insert(2, "std_Predicate", after_preprocess, True)
#     with open('finalfile.csv', 'w') as f:
#         #print(df.head())
#         df.to_csv(f)
#         f.close()

def process():
    df =pd.read_csv('./firstdraf.csv')
    predicates = df['Predicate']
    wnl = WordNetLemmatizer()
    after_preprocess =[]
    for p in predicates:
        res = ""
        tmp = pos_tag(p.split(" ")[1:])
        for t in tmp:
            if t[1]=='IN': continue
            if t[1]=='RB': continue
            if t[1]=='JJ': continue
            res+=wnl.lemmatize(t[0].lower(),'v')+" "
        after_preprocess.append(res)
    df.insert(2, "std_Predicate", after_preprocess, True)
    with open('finalfile.csv', 'w') as f:
        df.to_csv(f)
        f.close()
    
def main():
    testSVOs()
    process()
    

if __name__ == "__main__":
    main()
        


# In[ ]:





# In[ ]:




