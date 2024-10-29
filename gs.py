# from pygooglenews import GoogleNews
# gn = GoogleNews()
# search= gn.search('goldman sachs')
# print(search)

from requests_html import HTMLSession
import pandas as pd

c = 2000
lst=[]
for i in range(2000,2023) :
    url1 = 'https://news.google.com/rss/search?q=goldman-sachs-stocks-' + str(i)
    s= HTMLSession()
    r1=s.get(url1)
    for title in r1.html.find('title'):
        t1=title.text
        lst.append(t1)

#print(lst)

from transformers import pipeline

from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

model = BertForSequenceClassification.from_pretrained("ahmedrachid/FinancialBERT-Sentiment-Analysis",num_labels=3)
tokenizer = BertTokenizer.from_pretrained("ahmedrachid/FinancialBERT-Sentiment-Analysis")

nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

n = len(lst)
lst2 = []
for i in range(1,n) :
    v = lst[i]
    results = nlp(v)
    # print(v)
    # print(results)
    if results[0]['label'] == 'neutral':
        lst2.append(0)
    elif results[0]['label'] == 'positive':
        lst2.append(1)
    else :
        lst2.append(-1)

df = pd.DataFrame(list(zip(lst,lst2)), columns = ['news','sentiment_Analysis'])

#print(df)
url='https://drive.google.com/file/d/1xm5ztb1koYDoTu9PfVsJLfkqfGC8fJ89/view?usp=sharing'
url='https://drive.google.com/uc?id=' + url.split('/')[-2]
#df = pd.read_csv(url)
df2= pd.read_csv(url)
#print(df2)
l = min(len(df),len(df2))

df3 = df2.tail(l)
df3=df3.iloc[::-1]
print(df3)
print(len(df))

ind = []
for i in range(l) :
    ind.append(i)
df3.reset_index(inplace = True, drop = True)
print(df3)

final = pd.concat([df3,df], axis = 1)
print(final)

final.to_csv("C:\\Users\\varsh\\OneDrive\\Documents\\GitHub\\hackutd\\outputgs.csv")

    

# c = pipeline("sentiment-analysis")
# res = c(v)
# print(res)