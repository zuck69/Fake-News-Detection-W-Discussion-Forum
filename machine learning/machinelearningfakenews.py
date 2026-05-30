import numpy as np
import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

data1 = pd.read_csv("C:\\Users\\Asus\\College\\Final Year Project\\FND\\machine learning\\WELFake_Dataset.csv")

# Preprocessing
def wordopt(text):
    if isinstance(text, float):  # Check if the value is a float
        return str(text)  # Convert float to string
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

data1['text'] = data1['text'].apply(wordopt)

# Splitting data
data = data1.dropna()
X = data['text']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Vectorization
vectorization = TfidfVectorizer()
Xv_train = vectorization.fit_transform(X_train)
Xv_test = vectorization.transform(X_test)

# Save the vectorizer
joblib.dump(vectorization, 'tfidf_vectorizer.pkl')

print("Vecotorization done")

# Training models
LR = LogisticRegression()
LR.fit(Xv_train, y_train)
joblib.dump(LR, 'logistic_regression_model.pkl')

print("Logistic Regression Done")

DT = DecisionTreeClassifier()
DT.fit(Xv_train, y_train)
joblib.dump(DT, 'decision_tree_model.pkl')

print("Decision tree Done")

GB = GradientBoostingClassifier(random_state=0)
GB.fit(Xv_train, y_train)
joblib.dump(GB, 'gradient_boosting_model.pkl')

print("Gradient Boosting Done")

RF = RandomForestClassifier(random_state=0)
RF.fit(Xv_train, y_train)
joblib.dump(RF, 'random_forest_model.pkl')

print("Random Forest Done")

def manual_testing(news):
    # Load saved models
    LR = joblib.load('logistic_regression_model.pkl')
    DT = joblib.load('decision_tree_model.pkl')
    GB = joblib.load('gradient_boosting_model.pkl')
    RF = joblib.load('random_forest_model.pkl')
    vectorization = joblib.load('tfidf_vectorizer.pkl')

    # Preprocess input news
    news = wordopt(news)

    # Vectorize input news
    news_vectorized = vectorization.transform([news])

    # Make predictions
    pred_LR = LR.predict(news_vectorized)
    pred_DT = DT.predict(news_vectorized)
    pred_GB = GB.predict(news_vectorized)
    pred_RF = RF.predict(news_vectorized)

    return {
        "Logistic Regression Prediction": pred_LR[0],
        "Decision Tree Prediction": pred_DT[0],
        "Gradient Boosting Prediction": pred_GB[0],
        "Random Forest Prediction": pred_RF[0]
    }

print(manual_testing("""The president of Columbia University announced her resignation Wednesday after a little more than a year on the job, following months of criticism over protests on the Manhattan campus over the war in Gaza.

Nemat “Minouche” Shafik had been criticized by anti-war protesters and by House Republicans in Congress, but for different reasons.

In a letter to the Columbia community, Shafik said that while she was president "we have made progress in a number of important areas."

"However, it has also been a period of turmoil where it has been difficult to overcome divergent views across our community," she said. "This period has taken a considerable toll on my family, as it has for others in our community."

israel hamas conflict nypd columbia university hamilton hall entry
NYPD officers in riot gear enter a building at Columbia University on April 30.
Kena Betancur / AFP - Getty Images file
Shafik, an economist who became president of the Ivy League school in July 2023, asked the New York Police Department twice to clear encampments set up this spring by protesters in what demonstrators said was an act of solidarity with Palestinians.

After the first encampment on the Manhattan campus was cleared, a second one grew. Protesters took control of Hamilton Hall, and the NYPD cleared it and the encampment at the request of the university. In May, some students gathered outside Shafik’s house to protest and scream, typically a ritual held during finals.

In April, Shafik appeared before a House committee and faced questions about her handling of antisemitism on campus.

The resignation is effective Wednesday, Shafik wrote in the letter.

"Over the summer, I have been able to reflect and have decided that my moving on at this point would best enable Columbia to traverse the challenges ahead," Shafik wrote. "I am making this announcement now so that new leadership can be in place before the new term begins."

Columbia's Board of Trustees said in a statement that it "regretfully accepts Minouche Shafik’s decision to step down as president of the University."

A school leader with direct knowledge of the situation said the intense criticism and tension on campus amid the protests had become too much for Shafik to bear when it started to affect her family.

The person said they found the announcement unexpected despite the months of tensions at Columbia.

Katrina Armstrong was named interim president. She is CEO of the Columbia University Irving Medical Center and leads Columbia's health and biomedical sciences campus.

"With optimism and resolve, let us move forward together, embracing the opportunity to renew our vision and strengthen our community," Armstrong wrote in a letter about being named interim president.

The university was already anticipating a tough semester when classes resume next month, but it is now concerned about entering the new school year with new leadership who many faculty and students do not know, the school leader said. 

Columbia will focus on developing plans to be “proactive instead of reactive” this semester in its response to the protest movement and any lingering tension between the school and its trustees, who see the campus as too left-leaning, the school leader said.

The student protest group Columbia Students for Justice in Palestine welcomed the resignation on X. It had called on Shafik to resign.

"After months of chanting ‘Minouche Shafik you can’t hide’ she finally got the memo," the group said. "To be clear, any future president who does not pay heed to the Columbia student body’s overwhelming demand for divestment will end up exactly as President Shafik did."

Protests erupted at college campuses across the U.S. following the Oct. 7 terrorist attacks by Hamas against Israeli civilians, in which 1,200 people were killed and more than 200 hostages were taken, and during the subsequent war Israel launched against Hamas in Gaza. Since Oct. 7, over 39,900 people have been killed in Gaza, according to the Palestinian Ministry of Health.

Many protest groups called for their schools to divest from financial support of Israel, including those at Columbia demonstrations.

Elisha Baker, a junior at Columbia and leader in its group of Jewish students, has said that he felt unsafe on campus during the protests. He called Shafik's resignation "big news," but added: “This is about the leadership and governance of Columbia University.”

“The only thing that matters now is what happens next,” Baker, 21, said in a phone interview Wednesday. “I hope that Interim President Armstrong will assert strong leadership to keep Jewish students and all students safe, and to restore the values and the integrity of Columbia.”

Shafik wrote in the letter announcing her resignation that she holds as dear values that she said are Columbia’s values, which include free speech, openness to new ideas “and zero tolerance for discrimination of any kind.”

“Even as tension, division, and politicization have disrupted our campus over the last year, our core mission and values endure and will continue to guide us in meeting the challenges ahead,” Shafik wrote.

“I have tried to navigate a path that upholds academic principles and treats everyone with fairness and compassion. It has been distressing — for the community, for me as president and on a personal level — to find myself, colleagues, and students the subject of threats and abuse,” she wrote.

Heads of some universities became targeted by Republican lawmakers who alleged that demonstrations on college campuses were antisemitic.

University of Pennsylvania President Liz Magill resigned in December after she was criticized by the White House, lawmakers and others after she appeared to dodge a question at a congressional hearing about campus antisemitism.

Harvard University's president, Claudine Gay, resigned around a month later, in early January.

Rep. Elise Stefanik, R-N.Y., who had celebrated the earlier resignations, said in a statement Wednesday night: “Three down, so many to go.”"""))

print(manual_testing("""
On Friday, it was revealed that former Milwaukee Sheriff David Clarke, who was being considered for Homeland Security Secretary in Donald Trump s administration, has an email scandal of his own.In January, there was a brief run-in on a plane between Clarke and fellow passenger Dan Black, who he later had detained by the police for no reason whatsoever, except that maybe his feelings were hurt. Clarke messaged the police to stop Black after he deplaned, and now, a search warrant has been executed by the FBI to see the exchanges.Clarke is calling it fake news even though copies of the search warrant are on the Internet. I am UNINTIMIDATED by lib media attempts to smear and discredit me with their FAKE NEWS reports designed to silence me,  the former sheriff tweeted.  I will continue to poke them in the eye with a sharp stick and bitch slap these scum bags til they get it. I have been attacked by better people than them #MAGA I am UNINTIMIDATED by lib media attempts to smear and discredit me with their FAKE NEWS reports designed to silence me. I will continue to poke them in the eye with a sharp stick and bitch slap these scum bags til they get it. I have been attacked by better people than them #MAGA pic.twitter.com/XtZW5PdU2b  David A. Clarke, Jr. (@SheriffClarke) December 30, 2017He didn t stop there.BREAKING NEWS! When LYING LIB MEDIA makes up FAKE NEWS to smear me, the ANTIDOTE is go right at them. Punch them in the nose & MAKE THEM TASTE THEIR OWN BLOOD. Nothing gets a bully like LYING LIB MEDIA S attention better than to give them a taste of their own blood #neverbackdown pic.twitter.com/T2NY2psHCR  David A. Clarke, Jr. (@SheriffClarke) December 30, 2017The internet called him out.This is your local newspaper and that search warrant isn t fake, and just because the chose not to file charges at the time doesn t mean they won t! Especially if you continue to lie. Months after decision not to charge Clarke, email search warrant filed https://t.co/zcbyc4Wp5b  KeithLeBlanc (@KeithLeBlanc63) December 30, 2017I just hope the rest of the Village People aren t implicated.  Kirk Ketchum (@kirkketchum) December 30, 2017Slaw, baked potatoes, or French fries? pic.twitter.com/fWfXsZupxy  ALT- Immigration   (@ALT_uscis) December 30, 2017pic.twitter.com/ymsOBLjfxU  Pendulum Swinger (@PendulumSwngr) December 30, 2017you called your police friends to stand up for you when someone made fun of your hat  Chris Jackson (@ChrisCJackson) December 30, 2017Is it me, with this masterful pshop of your hat, which I seem to never tire of. I think it s the steely resolve in your one visible eye pic.twitter.com/dWr5k8ZEZV  Chris Mohney (@chrismohney) December 30, 2017Are you indicating with your fingers how many people died in your jail? I think you re a few fingers short, dipshit  Ike Barinholtz (@ikebarinholtz) December 30, 2017ROFL. Internet tough guy with fake flair. pic.twitter.com/ulCFddhkdy  KellMeCrazy (@Kel_MoonFace) December 30, 2017You re so edgy, buddy.  Mrs. SMH (@MRSSMH2) December 30, 2017Is his break over at Applebees?  Aaron (@feltrrr2) December 30, 2017Are you trying to earn your  still relevant  badge?  CircusRebel (@CircusDrew) December 30, 2017make sure to hydrate, drink lots of water. It s rumored that prisoners can be denied water by prison officials.  Robert Klinc (@RobertKlinc1) December 30, 2017Terrill Thomas, the 38-year-old black man who died of thirst in Clarke s Milwaukee County Jail cell this April, was a victim of homicide. We just thought we should point that out. It can t be repeated enough.Photo by Spencer Platt/Getty Images.
"""))

print(manual_testing("""
Recent studies have shown that drinking lemon juice can completely cure all types of cancer. This miraculous beverage has been proven to eradicate cancer cells and is a more effective treatment than conventional therapies. Experts recommend consuming at least one liter of lemon juice daily for best results"""))

manual_testing("""   
House Intelligence Committee Chairman Devin Nunes is going to have a bad day. He s been under the assumption, like many of us, that the Christopher Steele-dossier was what prompted the Russia investigation so he s been lashing out at the Department of Justice and the FBI in order to protect Trump. As it happens, the dossier is not what started the investigation, according to documents obtained by the New York Times.Former Trump campaign adviser George Papadopoulos was drunk in a wine bar when he revealed knowledge of Russian opposition research on Hillary Clinton.On top of that, Papadopoulos wasn t just a covfefe boy for Trump, as his administration has alleged. He had a much larger role, but none so damning as being a drunken fool in a wine bar. Coffee boys don t help to arrange a New York meeting between Trump and President Abdel Fattah el-Sisi of Egypt two months before the election. It was known before that the former aide set up meetings with world leaders for Trump, but team Trump ran with him being merely a coffee boy.In May 2016, Papadopoulos revealed to Australian diplomat Alexander Downer that Russian officials were shopping around possible dirt on then-Democratic presidential nominee Hillary Clinton. Exactly how much Mr. Papadopoulos said that night at the Kensington Wine Rooms with the Australian, Alexander Downer, is unclear, the report states. But two months later, when leaked Democratic emails began appearing online, Australian officials passed the information about Mr. Papadopoulos to their American counterparts, according to four current and former American and foreign officials with direct knowledge of the Australians role. Papadopoulos pleaded guilty to lying to the F.B.I. and is now a cooperating witness with Special Counsel Robert Mueller s team.This isn t a presidency. It s a badly scripted reality TV show.Photo by Win McNamee/Getty Images.
""")

print(manual_testing("""

On Christmas day, Donald Trump announced that he would  be back to work  the following day, but he is golfing for the fourth day in a row. The former reality show star blasted former President Barack Obama for playing golf and now Trump is on track to outpace the number of golf games his predecessor played.Updated my tracker of Trump s appearances at Trump properties.71 rounds of golf including today s. At this pace, he ll pass Obama s first-term total by July 24 next year. https://t.co/Fg7VacxRtJ pic.twitter.com/5gEMcjQTbH  Philip Bump (@pbump) December 29, 2017 That makes what a Washington Post reporter discovered on Trump s website really weird, but everything about this administration is bizarre AF. The coding contained a reference to Obama and golf:  Unlike Obama, we are working to fix the problem   and not on the golf course.  However, the coding wasn t done correctly.The website of Donald Trump, who has spent several days in a row at the golf course, is coded to serve up the following message in the event of an internal server error: https://t.co/zrWpyMXRcz pic.twitter.com/wiQSQNNzw0  Christopher Ingraham (@_cingraham) December 28, 2017That snippet of code appears to be on all https://t.co/dkhw0AlHB4 pages, which the footer says is paid for by the RNC? pic.twitter.com/oaZDT126B3  Christopher Ingraham (@_cingraham) December 28, 2017It s also all over https://t.co/ayBlGmk65Z. As others have noted in this thread, this is weird code and it s not clear it would ever actually display, but who knows.  Christopher Ingraham (@_cingraham) December 28, 2017After the coding was called out, the reference to Obama was deleted.UPDATE: The golf error message has been removed from the Trump and GOP websites. They also fixed the javascript  =  vs  ==  problem. Still not clear when these messages would actually display, since the actual 404 (and presumably 500) page displays a different message pic.twitter.com/Z7dmyQ5smy  Christopher Ingraham (@_cingraham) December 29, 2017That suggests someone at either RNC or the Trump admin is sensitive enough to Trump s golf problem to make this issue go away quickly once people noticed. You have no idea how much I d love to see the email exchange that led us here.  Christopher Ingraham (@_cingraham) December 29, 2017 The code was f-cked up.The best part about this is that they are using the  =  (assignment) operator which means that bit of code will never get run. If you look a few lines up  errorCode  will always be  404          (@tw1trsux) December 28, 2017trump s coders can t code. Nobody is surprised.  Tim Peterson (@timrpeterson) December 28, 2017Donald Trump is obsessed with Obama that his name was even in the coding of his website while he played golf again.Photo by Joe Raedle/Getty Images.
"""))