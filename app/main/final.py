from text_classification import classification
from sentiment import text_sentiment
from key import text_keywords
from summary import summarize
from nltk.tokenize import sent_tokenize
from sentiment import text_sentiment
def text_final(text):
        
        
        category= classification(text)
        #print ("Classify: ",classify)
        
        sentlist=[]
        sent_tokenize_list = sent_tokenize(text)

        #keyword=text_keywords(text)
        #print ("\nKeywors: ",keyw)
        summa= summarize(text,4)
        #print ("\nSumma: ",summa)
        possen=""
        negsen=""

        for senten in sent_tokenize_list:
                senti = text_sentiment(senten)
                if senti == 'pos':
                        possen=possen+" "+senten
                else:
                        negsen= negsen+" "+senten
        
        
        #print negsen        
        possum = summarize(possen,2)
        negsum = summarize(negsen,2)
                        
        #print '\n positive is ',possum
        #print '\n negative is ',negsum    
        
        return category,summa,possum,negsum
