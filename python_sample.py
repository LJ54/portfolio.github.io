#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 12:19:03 2020

@author: hsulong-jing
"""
#PART A: 爬蟲
import requests
from requests_html import HTML
import sqlite3

def fetch(url):
    response = requests.get(url)
    response = requests.get(url, cookies={'over18': '1'})  # 一直向 server 回答滿 18 歲了 !
    return response

def parse_article_entries(doc):
    html = HTML(html=doc)
    post_entries = html.find('div.r-ent')
    return post_entries

def parse_article_meta(ent):
    ''' Step-3 (revised): parse the metadata in article entry '''
    # 基本要素都還在
    meta = {
        'title': ent.find('div.title', first=True).text,
        'push': ent.find('div.nrec', first=True).text,
        'date': ent.find('div.date', first=True).text,
    }

    try:
        # 正常狀況取得資料
        meta['author'] = ent.find('div.author', first=True).text
        meta['link'] = ent.find('div.title > a', first=True).attrs['href']
    except AttributeError:
        # 但碰上文章被刪除時，就沒有辦法像原本的方法取得 作者 跟 連結
        if '(本文已被刪除)' in meta['title']:
            # e.g., "(本文已被刪除) [haudai]"
            match_author = re.search('\[(\w*)\]', meta['title'])
            if match_author:
                meta['author'] = match_author.group(1)
        elif re.search('已被\w*刪除', meta['title']):
            # e.g., "(已被cappa刪除) <edisonchu> op"
            match_author = re.search('\<(\w*)\>', meta['title'])
            if match_author:
                meta['author'] = match_author.group(1)
    return meta

    # 最終仍回傳統一的 dict() 形式 paired data
    return meta

# Part B: 搜尋 ＆ 分析基本資料
conn = sqlite3.connect('1.basics.sqlite')
cur = conn.cursor()
cur.execute('DROP TABLE IF EXISTS basics')
cur.execute('CREATE TABLE basics (date INTEGER, category TEXT, title TEXT, push INTEGER, link TEXT)')

import re 
page_list= list(range(1, 10))
search_endpoint_url = 'https://www.ptt.cc/bbs/TaiwanDrama/search'
for x in page_list:
        resp = requests.get(search_endpoint_url, params={'q': '我們與惡的距離', 'page': x})
        post_entries = parse_article_entries(resp.text)  # step-2
        for entry in post_entries:
            meta = parse_article_meta(entry)
            date = meta['date']
            push = meta['push']
            if push == '爆':
                push = 100
            title = meta['title']
            category = title [0:4]
            title = title [4:]
            link = meta['link']
            link = 'https://www.ptt.cc'+link
            if category == '[LIV':
                continue
            elif category == '[ANS':
                continue
            else:
                cur.execute('INSERT INTO basics (date, category, title, push, link) VALUES (?,?,?,?,?)', 
                            (date, category, title, push, link))
                print('PartB:', date, category, title, push, link)
                conn.commit()


#PART C: 取內文、留言
import requests
from bs4 import BeautifulSoup


def checkformat(soup, class_tag, data, index, link):
    # 避免有些文章會被使用者自行刪除 標題列 時間  之類......
    try:
        content = soup.select(class_tag)[index].text
    except Exception as e:
        print('checkformat error URL', link)
        # print 'checkformat:',str(e)
        content = "no " + data
    return content


#所要擷取的網站網址
with open('2.message.txt','w') as f:
    with open('2.content.txt', 'w') as u:
        cur.execute('SELECT link, date, category, title, push FROM basics')
        for row in cur:
            url = row[0]
            posted_date = row[1]
            category = row[2]
            posted_title = row [3]
            push = row [4]
            response = requests.get(url)
    #將原始碼做整理
            soup = BeautifulSoup(response.text, 'lxml')
            #print(soup)
    #使用find_all()找尋特定目標
            articles = soup.find_all('div', 'push')
            content = soup.find(id="main-content").text
            target_content = u'※ 發信站: 批踢踢實業坊(ptt.cc),'
        #去除掉 target_content
            content = content.split(target_content)
            #print(content)
            date = checkformat(soup, '.article-meta-value', 'date', 3, url)
            #print(date)
            title = checkformat(soup, '.article-meta-value', 'title', 2, url)
            content = content[0].split(date)
            #print(content)
        #去除掉文末 --
            main_content = content[1].replace('--', '')
        #印出內文
            print('Part C (content):', main_content)
            u.write("Date:"+posted_date + "\n")
            u.write("Category:"+category+ "\n")
            u.write("Title:"+ posted_title + "\n")
            u.write("Push:"+str(push)+ "\n")
            u.write(main_content + "\n")
            
            
    #寫入檔案中
            for article in articles:
                
            #去除掉冒號和左右的空白
                messages = article.find('span','f3 push-content').getText().replace(':','').strip()
                #print(messages)
                f.write("Date:"+ posted_date + "\n")
                f.write(messages + "\n")
                print('Part C (message):', messages)
            

f.close()
u.close()
cur.close 

#Part D: 尋找關鍵議題
from ckiptagger import construct_dictionary, WS, POS

ws = WS("./data")
pos = POS("./data")

content_dict={}
content_list= []

word_to_weight = {
            "我們與惡的距離": 1,
            "劇情":1,"假掰":1,"觀後感":1,"應思聰":1, "無雷":1, "首播":1,'事不關己':1, '網路霸凌':1,
            '共犯':1, '凌霸':1, '圍觀':1, '酸民':1, '設身處地':1, '刁民':1, '群眾':1,
            '旁觀者':1,'網民':1, '感同身受':1, '同理心':1, '霸凌':1, '鄉民':1, '冤殺':1, '冤枉':1, '冤罪':1, '冤錯':1,
            '判定':1, '判死':1, '司法':1, '司法史':1, '司法官':1, '司法權':1, 
            '安樂死':1, '安樂':1, '審判長':1, '審定':1, '審查':1, '審查會':1, '審視':1, '懲罰':1, '懲罰性':1,
            '死囚犯':1, '活下來':1, '死刑庭':1, '辯護人':1, '送審':1, '死刑案':1, '裁決':1, '以暴制暴':1, '黃致豪':1 ,'冤案':1,
            '無罪':1, '減刑':1, '開庭':1, '活下去的權利':1, '罪犯':1, '起訴':1, '無期徒刑':1, '王赦':1, '法律人':1, 
            '處死':1, '判決':1, '殺人案':1, '法官':1, '吳慷仁':1, '法律':1, '處死':1, '判決':1, '殺人案':1, 
            '法官':1, '吳慷仁':1, '法律':1, '干預':1, '新聞自由':1, '社會亂源':1, '先驅報':1, '報導':1, '爆炸案':1, '追殺':1,
            '媒體人':1, '點擊率':1, '點閱率':1, '主播':1, '新聞部':1, '新聞報導':1, '兇手':1, '兇殘':1, '兇殺':1, '兇殺案':1, '槍擊':1, '持刀':1, 
            '有期徒刑':1, '殺童案':1, '燈泡':1, '十惡不赦':1, '殺人魔':1, '無差別殺人事件':1, '幼稚園':1, '鄭捷':1, 
            '殺人':1, '鄭捷案':1, '判若兩人':1, '犯罪動機':1, '喪禮':1, '憂鬱':1,
            '犯罪率':1, '身心科':1, '辱罵':1, '鎮靜劑':1, '應思悅':1, '汙名化':1, '污名化':1, '焦慮症':1, '犯罪人':1, 
            '療程':1, '看醫生':1, '神經病':1, '社會化':1, '焦慮':1, '鎮靜室':1, '心理治療':1, '躁鬱症':1,
            '歧視':1, '殺人犯':1, '受刑人':1, '精障者':1, '療養院':1, '憂鬱症':1, '李曉文':1, '發病':1, '加害人家屬':1, 
            '精神病':1, '生病':1, '治療':1, '精神科':1, '李大芝':1, '失調症':1, '曾沛慈':1, '應思聰':1, '林哲熹':1, 
            '大芝':1, '李曉明':1, '悲痛欲絕':1, '冷冰冰':1, '脆弱':1, '酗酒':1, '釋懷':1, '天晴':1, '被害者家屬':1, '賈靜雯':1, 
            '宋喬安':1, '受害':1, '溫昇豪':1, '劉昭國':1,
    }

delimiter_set = {"，", "。", "　", "；", "、", "？", "：", "（", "）", "「", "」", "》", "《", "！", "\u3000","/", "\"", '…', '【', '】' , '』','『', '...', '～', '｜', '-', '+','~','・',}

def fhandle(x): #我只要內容
    fhandle=open(x, mode='r', encoding='utf-8-sig')
    y=list(enumerate(fhandle))
    dictionary = construct_dictionary(word_to_weight)
    
    
    for line in y:
        line= list(line)
        line= (line[1])  
        if line == '\n':
            continue
        elif line.startswith("Date:"):
            continue
        elif line.startswith("Category:"):
            continue
        elif line.startswith("Title:"):
            continue
        elif line.startswith("Push:"):
            continue
        elif line.startswith("https"):
            continue
        else:
            list_content= list([line],)
            #print(list_content)
             
            word_sentence_list = ws(list_content, segment_delimiter_set = delimiter_set, sentence_segmentation=True, recommend_dictionary=dictionary)
            print (word_sentence_list)
            for words in word_sentence_list:
                for word in words:
                    content_dict[word]= content_dict.get(word, 0)+1
                    content_list= list()

def fhandle2(x):
    with open(x,'w') as f:
        for key, value in content_dict.items():
            if value >0:
                content_list.append((value, key))
                content_list.sort(reverse=True)
        for key, value in content_list:
            content_kv= (key, value)
            content_kv= str(content_kv)
            f.write(content_kv + "\n")
    f.close()



#fhandle('2.content.txt')
#fhandle2('3.content_analysis.txt')
#print('Part D: contents done')
content_dict={}
content_list= []
#fhandle('2.message.txt')
#fhandle2('3.message_analysis.txt')
#print('Part D: message done')

#Part E 計算＆計算數量（content)

alist= ('酸民', '設身處地', '刁民', '群眾','旁觀者','網民', '感同身受', '同理心', '霸凌', '鄉民')
blist= ('罪犯', '起訴', '無期徒刑', '王赦', '法律人', '處死', '判決', '殺人案', '法官', '法律')
clist= ('先驅報', '報導', '爆炸案', '追殺', '媒體人', '點擊率', '點閱率','主播', '新聞部', '新聞報導')
dlist= ('殺童案', '燈泡', '十惡不赦', '殺人魔', '無差別殺人事件', '幼稚園', '鄭捷', '殺人', '鄭捷案', '鄭捷')
elist= ('加害人家屬', '精神病', '生病', '治療', '精神科', '李大芝', '失調症', '應思聰','大芝', '李曉明')
flist= ('悲痛欲絕', '冷冰冰', '脆弱', '酗酒', '釋懷', '天晴', '被害者家屬', '宋喬安', '受害', '劉昭國' ) 

#Part E-1 計算＆計算數量（content)
conn = sqlite3.connect('4.content_analysis.sqlite')
cur = conn.cursor()
cur.execute('DROP TABLE IF EXISTS content_analysis')
cur.execute('CREATE TABLE content_analysis (Date TEXT, Category TEXT, Push TEXT, Atype INTEGER, Btype INTEGER, Ctype INTEGER, Dtype INTEGER, Etype INTEGER, Ftype INTEGER)')
    
with open('2.content.txt', 'r') as f:
        lines= f.readlines()
        lines= [line.replace('\n', '') for line in lines]
        liness= [line.replace('Date:', '\n Date:') for line in lines]
        #print(liness)
with open('short content.txt', 'w') as f:
             f.writelines(lines)
                        
with open('short content.txt', 'w') as f:
            f.writelines(liness)
                    
with open('short content.txt') as f:
            sentencelist = [line.rstrip('\n') for line in f]
            new_sentencelist= []
            for sentence in sentencelist:
                if sentence is '':
                    continue
                else:
                    basics= re.findall('Date:.+Push:..', sentence)
                    basics = basics[0]
                    Date = re.findall('Date:([0-9.]+/[0-9.]+)',basics)
                    Category = re.findall('Category:(.+)Title', basics)
                    Push = re.findall('Push:([0-9.]+)', basics)
                    if Push == []:
                        Push = [0,]
                    print(Date, Category, Push)
                    cur.execute('INSERT INTO content_analysis (Date, Category, Push) VALUES (?,?,?)', (Date[0], Category[0], Push[0]))
                    conn.commit()
                    sentence= sentence.replace(basics, '')
                    new_sentencelist.append(sentence)
dictionary = construct_dictionary(word_to_weight)
       #             print(sentencelist)
word_sentence_list = ws(new_sentencelist, sentence_segmentation=True, recommend_dictionary=dictionary)
                    #print(word_sentence_list)
print('Part E-1 check', len(word_sentence_list))

counta= 0
countb= 0
countc= 0
countd= 0 
counte= 0
countf =0
zero =0
Atype =0
Btype =0
Ctype =0
Dtype =0
Etype =0
Ftype =0 
maxAtype =0
maxBtype =0
maxCtype =0
maxDtype =0
maxEtype =0
maxFtype =0                      

for sentence in word_sentence_list:
    for word in sentence:
        if word in alist:
            counta+=1
        if word in blist:
            countb+=1
        if word in clist:
            countc+=1
        if word in dlist:
            countd+=1
        if word in elist:
            counte+=1
        if word in flist:
            countf+=1
    if max(counta, countb, countc, countd, counte, countf) == 0:
            zero +=1
            cur.execute('REPLACE INTO content_analysis (Atype, Btype, Ctype, Dtype, Etype, Ftype) VALUES (?,?,?,?,?,?)', ('0','0', '0','0','0','0'))
            conn.commit()
            continue
    if max(counta, countb, countc, countd, counte, countf) is counta:
            Atype +=1
            maxAtype+=1
    if max(counta, countb, countc, countd, counte, countf) is countb:
            Btype +=1
            maxBtype +=1
    if max(counta, countb, countc, countd, counte, countf) is countc:
            Ctype +=1
            maxCtype +=1
    if max(counta, countb, countc, countd, counte, countf) is countd:
            Dtype +=1
            maxDtype +=1
    if max(counta, countb, countc, countd, counte, countf) is counte:
            Etype +=1
            maxEtype +=1
    if max(counta, countb, countc, countd, counte, countf) is countf:
            Ftype +=1
            maxFtype +=1
                        #重複的也都算
    if len(sentence)>1:
            #print (sentence)
            print ('Part E: contents', counta, countb, countc, countd, counte, countf, '*****',
                   Atype, Btype, Ctype, Dtype, Etype, Ftype, '******', maxAtype, maxBtype, maxCtype, maxDtype, maxEtype, maxFtype)  
                            #這是每一個的 max
    cur.execute('REPLACE INTO content_analysis (Atype, Btype, Ctype, Dtype, Etype, Ftype) VALUES (?,?,?,?,?,?)', (Atype, Btype, Ctype, Dtype, Etype, Ftype))
    conn.commit()
                            
    counta = 0
    countb=0
    countc=0
    countd=0
    counte=0
    countf=0
    Atype =0
    Btype =0
    Ctype =0
    Dtype =0
    Etype =0
    Ftype =0
                        
print('A 網路酸民:', maxAtype, '\nB 律師:', maxBtype, '\nC 媒體:', maxCtype, '\nD 社會殺人事件:', maxDtype, '\nE 加害者: ', maxEtype, '\nF 被害者:', maxFtype, '\nzero match:', zero)
cur.close


#Part E-2: 分類 ＆ 計算數量 (message)
conn = sqlite3.connect('4.message_analysis.sqlite')
cur = conn.cursor()
cur.execute('DROP TABLE IF EXISTS message_analysis')
cur.execute('CREATE TABLE message_analysis (Date TEXT, Atype INTEGER, Btype INTEGER, Ctype INTEGER, Dtype INTEGER, Etype INTEGER, Ftype INTEGER)')

with open('2.message.txt', 'r') as f:
        lines= f.readlines()
lines= [line.replace('\n', '') for line in lines]
liness= [line.replace('Date:', '\n Date:') for line in lines]
    #print(lines)
    
with open('short content.txt', 'w') as f:
        f.writelines(lines)
        
with open('short content.txt', 'w') as f:
        f.writelines(liness)
    
with open('short content.txt') as f:
        sentencelist = [line.rstrip('\n') for line in f]
    
dictionary = construct_dictionary(word_to_weight)
    
word_sentence_list = ws(sentencelist, sentence_segmentation=True, recommend_dictionary=dictionary)
print(len(word_sentence_list))
    

counta= 0
countb= 0
countc= 0
countd= 0 
counte= 0
countf =0
zero =0
Atype =0
Btype =0
Ctype =0
Dtype =0
Etype =0
Ftype =0 
maxAtype =0
maxBtype =0
maxCtype =0
maxDtype =0
maxEtype =0
maxFtype =0    

for sentence in word_sentence_list:
        for word in sentence:
            if word in alist:
                counta+=1
            if word in blist:
                countb+=1
            if word in clist:
                countc+=1
            if word in dlist:
                countd+=1
            if word in elist:
                counte+=1
            if word in flist:
                countf+=1
        if max(counta, countb, countc, countd, counte, countf) == 0:
            zero +=1
            continue
        if max(counta, countb, countc, countd, counte, countf) is counta:
                Atype +=1
                maxAtype+=1
        if max(counta, countb, countc, countd, counte, countf) is countb:
                Btype +=1
                maxBtype +=1
        if max(counta, countb, countc, countd, counte, countf) is countc:
                Ctype +=1
                maxCtype +=1
        if max(counta, countb, countc, countd, counte, countf) is countd:
                Dtype +=1
                maxDtype +=1
        if max(counta, countb, countc, countd, counte, countf) is counte:
                Etype +=1
                maxEtype +=1
        if max(counta, countb, countc, countd, counte, countf) is countf:
                Ftype +=1
                maxFtype +=1
        #重複的也都算
        if len(sentence)>1:
            Date = sentence[2:3]
            Date = Date[0]
            print ('Part E: messages', Date, counta, countb, countc, countd, counte, countf, '*****',
                   Atype, Btype, Ctype, Dtype, Etype, Ftype)  
            #這是每一個的 max
            cur.execute('INSERT INTO message_analysis (Date, Atype, Btype, Ctype, Dtype, Etype, Ftype) VALUES (?,?,?,?,?,?,?)', (Date, Atype, Btype, Ctype, Dtype, Etype, Ftype))
            conn.commit()
            
        counta = 0
        countb=0
        countc=0
        countd=0
        counte=0
        countf=0
        Atype =0
        Btype =0
        Ctype =0
        Dtype =0
        Etype =0
        Ftype =0
        
print('A 網路酸民:', maxAtype, '\nB 律師:', maxBtype, '\nC 媒體:', maxCtype, '\nD 社會殺人事件:', 
                  maxDtype, '\nE 加害者: ', maxEtype, '\nF 被害者:', maxFtype, '\nzero match:', zero)
cur.close


#Part F: 計算文章的相關性
import os
import codecs
import nltk
import math
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def getText(txtFileName):
    file = codecs.open(txtFileName, 'r', encoding='utf-8', errors='replace')
    return file.read()
      
# Show results
def print_word_pos_article(word_article, pos_article):
    assert len(word_article) == len(pos_article)
    for word, pos in zip(word_article, pos_article):
        print(f"{word}({pos})", end="\u3000")
    print()
    return

#只留下名詞、動詞、形容詞
def extract_keywords(word_article, pos_article):
    assert len(word_article) == len(pos_article)
    
    keywords = list()
    for word, pos in zip(word_article, pos_article):
        if pos[0] == 'N' or pos[0] == 'V' or pos[0] == 'A' :
            keywords.append(word)

    return keywords
                    
def TF(keywords_article):
    freq = nltk.FreqDist(keywords_article)
    dictionary = {}
    for key in freq.keys():
        norm = freq[key]/float(len(keywords_article))
        dictionary[key] = norm
    return dictionary

# 計算關鍵字的IDF
# 參數 tf_corpus 為一字典的清單，清單中每一元素為一字典，記錄每篇文章關鍵字的TF值
def IDF(tf_corpus):
    def idf(TotalNumberOfDocuments, NumberOfDocumentsWithThisWord):
        return 1.0 + math.log(TotalNumberOfDocuments/NumberOfDocumentsWithThisWord)
    
    numDocuments = len(tf_corpus)
    uniqueWords = {}
    idfValues = {}
    for article in tf_corpus:
        for word in article.keys():
            if word not in uniqueWords:
                uniqueWords[word] = 1
            else:
                uniqueWords[word] += 1
                    
    for word in uniqueWords:
        idfValues[word] = idf(numDocuments, uniqueWords[word])
        
    return idfValues

#根據關鍵字筆畫順序建立 TF-IDF 矩陣
def TF_IDF(idf_keywords, tf_corpus):
    keywords = list(idf_keywords.keys())
    keywords.sort()
    tf_idf_matrix = list()
    for keyword in keywords:
        keyword_list = list()
        for tf_article in tf_corpus:
            tfv = tf_article[keyword] if keyword in tf_article else 0.0
            mul = idf_keywords[keyword] * tfv
            keyword_list.append(mul)
        tf_idf_matrix.append(keyword_list)
    return tf_idf_matrix




files = os.listdir("corpus")

# corpus 放置所有文章的內容
corpus = list()


#讀取目錄中所有檔案，建立 CKIP 所需的資料格式
for file in files:
    filename = ".//corpus//" + file
    text = getText(filename)
#   text = text.translate(text.maketrans(intab, outtab))
    corpus.append(text)

#斷詞切字    
word_corpus_list = ws(corpus, segment_delimiter_set = delimiter_set)
pos_corpus_list = pos(word_corpus_list, segment_delimiter_set = delimiter_set)

del ws
del pos

# 印出關鍵字和其詞性
for i, article in enumerate(corpus):
    print()
    print(f"'{article}'")
    print_word_pos_article(word_corpus_list[i],  pos_corpus_list[i])

# keywords_corpus_list 只放部分詞性的關鍵字
keyword_corpus_list = list()  

for i, article in enumerate(corpus):
    keywords_article = extract_keywords(word_corpus_list[i],  pos_corpus_list[i]) 
    keyword_corpus_list.append(keywords_article)
    
#tf_corpus包含每個字在每篇文章中的TF
tf_corpus = list()
for keywords_article in keyword_corpus_list:
    tf_corpus.append(TF(keywords_article))
    
idf_keywords = IDF(tf_corpus)
tf_idf_matrix = TF_IDF(idf_keywords, tf_corpus)
keyword_based_matrix = np.array(tf_idf_matrix)
doc_based_matrix = np.transpose(keyword_based_matrix)
keyword_similarity = cosine_similarity(keyword_based_matrix)
doc_similarity = cosine_similarity(doc_based_matrix)
