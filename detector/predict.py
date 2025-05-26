import pandas as pd
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from detector.utils import wordopt
from sklearn.model_selection import train_test_split

data_fake = pd.read_csv(r"C:\Users\sweth\fake_news_detection\Fake.csv")
data_true = pd.read_csv(r"C:\Users\sweth\fake_news_detection\True.csv")
data_fake.head()
data_true.tail()
data_fake["class"]=0
data_true['class']=1
data_fake.shape, data_true.shape

data_fake_manual_testing = data_fake.tail(10)
for i in range(23480,23470,-1):
    data_fake.drop([i],axis = 0, inplace = True)

    
data_true_manual_testing = data_true.tail(10)
for i in range(21416,21406,-1):
    data_true.drop([i],axis = 0, inplace = True)

data_fake.shape, data_true.shape

    
data_fake_manual_testing.loc[:, 'class']=0
data_true_manual_testing.loc[:, 'class']=1
data_fake_manual_testing.head(10)
data_true_manual_testing.head(10)
data_merge=pd.concat([data_fake, data_true], axis = 0)
data_merge.head(10)
data_merge.columns
data=data_merge.drop(['title','subject','date'], axis = 1)
data.isnull().sum() 
data = data.sample(frac = 1)
data.head()
data.reset_index(inplace = True)
data.drop(['index'], axis = 1, inplace = True)
data.columns
data.head()

import re
import string

def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)  # Remove text within brackets
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>+', '', text)  # Remove HTML tags
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  # Remove punctuation
    text = re.sub(r'\w*\d\w*', '', text)  # Remove words containing numbers
    return text

data['text'] = data['text'].apply(wordopt)
x = data['text']
y = data['class']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25)
vectorization = TfidfVectorizer(
    max_df=0.7,
    min_df=5,
    stop_words='english',
    ngram_range=(1,2)  # unigrams and bigrams
)
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)
# Assuming x_tra

# Initialize your models (make sure these are defined)
LR = LogisticRegression()
DT = DecisionTreeClassifier()
GB = GradientBoostingClassifier(random_state=0)
RF = RandomForestClassifier(random_state=0)


LR.fit(xv_train, y_train)
DT.fit(xv_train, y_train)
GB.fit(xv_train, y_train)
RF.fit(xv_train, y_train)
def output_lable(n):
    if n==0:
        return "Fake News"
    elif n==1:
        return "Not A Fake News"
    

def manual_testing(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)  # CLEANING STEP
    new_xv_test = vectorization.transform(new_def_test["text"])

    pred_RF = RF.predict(new_xv_test)
    prediction = pred_RF[0]

    if prediction == 0:
        overall_result = "Fake News"
    else:
        overall_result = "Not A Fake News"

    return overall_result
