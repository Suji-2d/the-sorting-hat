import streamlit as st
import numpy as np
import json
from PIL import Image
import pickle
import pandas as pd


image1 = Image.open('./data/images/hat-bg3.png')
image2 = Image.open('./data/images/hat-bg1.png')

st.write("""# The Sorting Hat
""")

st.image(image1)
with open('./data/TheBig5questions.json', 'r') as f:  
  questions = json.load(f)

st.write("""#### Ready to find out which **house** you belong!
##### Let me ask few questions to check your personality, OKAY! little lot of questions! 😅""")

st.markdown("""---""")

ansList=[]
options = ["disagree", "slightly disagree", "neutral", "slightly agree", "agree"]

for i in range(1,26):
   st.write(f"""
   #### {questions[str(i)]}
   """)
   ansList.append(st.radio("choose...",
        options,
        index=2,
        key=i,
        horizontal=True,
        label_visibility='collapsed',
))
st.text(" ")
st.text(" ")

def scoringForAns(ans):
    return options.index(ans) + 1

def score_scale_fun(X, scores_old_min, scores_old_max, scores_new_min=0, scores_new_max=1):  
  X = scores_new_max - ((scores_new_max - scores_new_min) * (scores_old_max - X) / (scores_old_max - scores_old_min))    
  return X  

cModel = pickle.load(open('houseModel.pkl','rb'))
scores=[]
if st.button("Submit"):
  newAnsList=[scoringForAns(a) for a in ansList]
  #print(f'user answers: {newAnsList}')
  extro = 7 + newAnsList[0] - newAnsList[1] + newAnsList[10] + newAnsList[20] - newAnsList[6]
  agree = 7 + newAnsList[2] + newAnsList[12] + newAnsList[18] - newAnsList[11] - newAnsList[21]
  consci = 7 + newAnsList[3] + newAnsList[13] + newAnsList[22] - newAnsList[4] - newAnsList[14]
  neurot= 19 + newAnsList[7] - newAnsList[5] - newAnsList[15] - newAnsList[16] - newAnsList[23]
  open = 1 + newAnsList[9] + newAnsList[17] + newAnsList[19] + newAnsList[24] - newAnsList[8]
  scores=[extro,agree,consci,neurot,open]
  newSocres = [score_scale_fun(s,0,20,0,50) for s in scores]
  print([newSocres])
  #print(f'old20: {scores},/n new50: {newSocres}')
  y_predict =pd.DataFrame(
    [newSocres],
    columns=['Extroversion', 'Agreeableness', 'Conscientiousness','Neuroticism','Openness']
    )
  print(y_predict)
  house_info={
    "Gryffindor":
"""Gryffindor house is where you would find the pluckiest and most daring students (there’s a reason the house symbol is the brave lion). The house colours are scarlet and gold, the common room lies up in Gryffindor Tower and the Head of House is Professor Minerva McGonagall.

If the Sorting Hat placed you here, you would have demonstrated qualities like courage, bravery and determination. Some of the wizarding world’s best and brightest belonged to this house – Harry Potter and Albus Dumbledore are just a couple that spring to mind!

If you are lucky enough to end up in Gryffindor, we imagine you’re the type of person who likes to stand up for the little guy, challenges authority, has a tendency to act first and think later, is known as a class clown and takes board games very seriously.""",
"Slytherin":
"""Slytherin house has an unfortunate reputation. While it is true that a lot of dark witches and wizards were sorted into Slytherin, not all who belong to this house are bad. In fact, there are a lot of excellent qualities the Sorting Hat looks for in potential Slytherins and Merlin himself even belonged to this misunderstood house!

The house colours for Slytherin are silver and emerald green and the emblem is a serpent. The Head of House is Professor Severus Snape, and the common room can be found down in the dungeons under the lake (which only adds to the Slytherin air of mystery).

If the Sorting Hat placed you in this noble house, then you are most likely ambitious, shrewd and possibly destined for greatness. We can imagine you’re the kind of person who is always one step ahead, has a dark sense of humour, thinks reputation is important, takes pride in their appearance and doesn’t let anyone see their soft side.""",
"Ravenclaw":
"""If you are looking for the brainiest students – you would find them in Ravenclaw (with a couple of notable exceptions like Hermione Granger). The Ravenclaw colours are blue and bronze, the emblem is an eagle, the Head of House is Professor Filius Flitwick and the common room sits at the top of Ravenclaw Tower behind an enchanted knocker.

The Sorting Hat would only put you in this house if you demonstrated excellent wisdom, wit and a skill for learning. Ravenclaws are often known for being quite eccentric and most of the great wizarding inventors and innovators have come from this house.

We can imagine that you would get to sit up in Ravenclaw Tower, while surveying the excellent views, if you’re the type of person who analyses everything, is an overachiever, can be described as away with the fairies, is not afraid to be an individual and has a head stuffed full of interesting facts.
""",
"Hufflepuff":
"""Hufflepuff is where you will find the most trustworthy and hardworking students. In fact, out of all the houses Hufflepuff has produced the least number of dark witches and wizards. The badger is the symbol of this house, the colours are yellow and black, the Head of House is Professor Pomona Sprout and the common room can be found near the kitchens in Hogwarts.

There is an idea that Hufflepuffs are the least clever of all Hogwarts students – but that is not true. Hufflepuffs are just the most humble of all the houses and don’t feel the need to shout about their achievements in the same way as the others.

If you were lucky enough to be sorted into this house, we can imagine you’re the type of person who has a strong moral compass, always works hard, is the most loyal friend, knows it is the taking part that counts and always has the best snacks.
"""}

  predicted_house=cModel.predict(y_predict)[0]
  st.write(f""" 
    ---
    # {predicted_house} !!!
  """)
  st.write(house_info[predicted_house])
  st.image(image2, caption="Make your house proud, YOUNG WIZARD!!")

st.write("""
  ---
  [Reference blog](https://ai.plainenglish.io/the-algorithm-behind-the-harry-potter-sorting-hat-45e41379929f) - Modified big five personality test and dataset. House info collected from [here](https://www.wizardingworld.com/features/hogwarts-house-meanings).
  
  """, )




