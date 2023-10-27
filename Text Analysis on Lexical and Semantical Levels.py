# Importing modules for NLP 
import re
import nltk
import spacy
nlp = spacy.load("en_core_web_sm")

# Importing ML modules for NER
import flair
from flair.data import Sentence 
from flair.models import SequenceTagger
tagger = SequenceTagger.load("flair/ner-english")

# Importing modules for data analysis and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

# Importing a module for time founding 
import time 

# A startpoint of the program 
start = time.time()

# Reading words classified by CEFR levels (from A1 to C1) from the files, creating lists of words without POS-tags
with open("Lexis_А1.txt", "rt", encoding = "utf-8") as file:
    text_A1 = file.read()
    text_A1 = re.sub(r"[\n]", " ", text_A1)
    words_with_tags_A1 = text_A1.split()
    words_without_tags_A1 = np.array([words_with_tags_A1[i] for i in range(len(words_with_tags_A1)) if i % 2 == 0])

with open("Lexis_А2.txt", "rt", encoding = "utf-8") as file:
    text_A2 = file.read()
    text_A2 = re.sub(r"[\n]", " ", text_A2)
    words_with_tags_A2 = text_A2.split()
    words_without_tags_A2 = np.array([words_with_tags_A2[i] for i in range(len(words_with_tags_A2)) if i % 2 == 0])

with open("Lexis_В1.txt", "rt", encoding = "utf-8") as file:
    text_B1 = file.read()
    text_B1 = re.sub(r"[\n]", " ", text_B1)
    words_with_tags_B1 = text_B1.split()
    words_without_tags_B1 = np.array([words_with_tags_B1[i] for i in range(len(words_with_tags_B1)) if i % 2 == 0])

with open("Lexis_В2.txt", "rt", encoding = "utf-8") as file:
    text_B2 = file.read()
    text_B2 = re.sub(r"[\n]", " ", text_B2)
    words_with_tags_B2 = text_B2.split()
    words_without_tags_B2 = np.array([words_with_tags_B2[i] for i in range(len(words_with_tags_B2)) if i % 2 == 0])

with open("Lexis_С1.txt", "rt", encoding = "utf-8") as file:
    text_C1 = file.read()
    text_C1 = re.sub(r"[\n]", " ", text_C1)
    words_with_tags_C1 = text_C1.split()
    words_without_tags_C1 = np.array([words_with_tags_C1[i] for i in range(len(words_with_tags_C1)) if i % 2 == 0])

# Creating a set with unique A1 lexis according to the CEFR
words_unique_A1_final = words_without_tags_A1

# Creating a set with unique A2 lexis according to the CEFR
words_unique_A2_final = set(words_without_tags_A2) - set(words_without_tags_A1)

# Creating a set with unique B1 lexis according to the CEFR
words_unique_B1_final = set(words_without_tags_B1) - set(words_without_tags_A1)
words_unique_B1_final = set(words_unique_B1_final) - set(words_without_tags_A2)

# Creating a set with unique B2 lexis according to the CEFR
words_unique_B2 = set(words_without_tags_B2) - set(words_without_tags_A1)
words_unique_B2_medium = set(words_unique_B2) - set(words_without_tags_A2)
words_unique_B2_final = set(words_unique_B2_medium) - set(words_without_tags_B1)

# Creating a set with unique C1 lexis according to the CEFR
words_unique_C1 = set(words_without_tags_C1) - set(words_without_tags_A1)
words_unique_C1_medium = set(words_unique_C1) - set(words_without_tags_A2)
words_unique_C1_pre_final= set(words_unique_C1_medium) - set(words_without_tags_B1)
words_unique_C1_final= set(words_unique_C1_pre_final) - set(words_without_tags_B2)

# Creating lists with unique A1-C1 lexis according to the CEFR
words_unique_A1_final = list(words_unique_A1_final)
words_unique_A2_final = list(words_unique_A2_final)
words_unique_B1_final = list(words_unique_B1_final)
words_unique_B2_final = list(words_unique_B2_final)
words_unique_C1_final = list(words_unique_C1_final)

# Reading collocations from files classified by CEFR levels (from A1 to C1)
with open("Collocations_А1.txt", "rt", encoding = "utf-8") as file:
    collocations_A1_without_tags = np.array([line.strip() for line in file])

with open("Collocations_А2.txt", "rt", encoding = "utf-8") as file:
    collocations_A2_without_tags = np.array([line.strip() for line in file])

with open("Collocations_В1.txt", "rt", encoding = "utf-8") as file:
    collocations_B1_without_tags = np.array([line.strip() for line in file])

with open("Collocations_В2.txt", "rt", encoding = "utf-8") as file:
    collocations_B2_without_tags = np.array([line.strip() for line in file])

with open("Collocations_С1.txt", "rt", encoding = "utf-8") as file:
    collocations_C1_without_tags = np.array([line.strip() for line in file])

# Reading words and colocations classified by topics according to the Oxford dictionary 
with open("Animals.txt", "rt", encoding = "utf-8") as file:
    file = file.read()
    file = re.sub(r"[\n]", " ", file)
    animals = file.split()
    animals = list(set(animals))

with open("Appearance.txt", "rt", encoding = "utf-8") as file:
    file = file.read()
    file = re.sub(r"[\n]", " ", file)
    appearance = file.split()
    appearance = list(set(appearance))

with open("Communication.txt", "rt", encoding = "utf-8") as file:
    file = file.read()
    file = re.sub(r"[\n]", " ", file)
    communication = file.split()
    communication = list(set(communication))

with open("Culture.txt", "rt", encoding = "utf-8") as file:
    file = file.read()
    file = re.sub(r"[\n]", " ", file)
    culture = file.split()
    culture = list(set(culture))

with open("Food and Drink.txt", "rt", encoding = "utf-8") as file:
    file = file.read()
    file = re.sub(r"[\n]", " ", file)
    food_and_drink = file.split()
    food_and_drink = list(set(food_and_drink))

with open("Functions.txt", "rt", encoding = "utf-8") as file:
    file = file.read()
    file = re.sub(r"[\n]", " ", file)
    functions = file.split()
    functions = list(set(functions))

with open("Health.txt", "rt", encoding = "utf-8") as file:
    file = file.read()
    file = re.sub(r"[\n]", " ", file)
    health = file.split()
    health = list(set(health))

with open("Homes and Buildings.txt", "rt", encoding = "utf-8") as file:
    file = file.read()
    file = re.sub(r"[\n]", " ", file)
    homes_and_buildings = file.split()
    homes_and_buildings = list(set(homes_and_buildings))

with open("Leisure.txt", "rt", encoding = "utf-8") as file:
    file = file.read()
    file = re.sub(r"[\n]", " ", file)
    leisure = file.split()
    leisure = list(set(leisure))

with open("Notions.txt", "rt", encoding = "utf-8") as file:
    file = file.read()
    file = re.sub(r"[\n]", " ", file)
    notions = file.split()
    notions = list(set(notions))

with open("People.txt", "rt", encoding = "utf-8") as file:
    file = file.read()
    file = re.sub(r"[\n]", " ", file)
    people = file.split()
    people = list(set(people))

with open("Politics and Society.txt", "rt", encoding = "utf-8") as file:
    file = file.read()
    file = re.sub(r"[\n]", " ", file)
    politics_and_society = file.split()
    politics_and_society = list(set(politics_and_society))

with open("Science and Technology.txt", "rt", encoding = "utf-8") as file:
    file = file.read()
    file = re.sub(r"[\n]", " ", file)
    science_and_technology = file.split()
    science_and_technology = list(set(science_and_technology))

with open("Sports.txt", "rt", encoding = "utf-8") as file:
    file = file.read()
    file = re.sub(r"[\n]", " ", file)
    sports = file.split()
    sports = list(set(sports))

with open("The Natural World.txt", "rt", encoding = "utf-8") as file:
    file = file.read()
    file = re.sub(r"[\n]", " ", file)
    the_natural_world = file.split()
    the_natural_world = list(set(the_natural_world))

with open("Time and Space.txt", "rt", encoding = "utf-8") as file:
    file = file.read()
    file = re.sub(r"[\n]", " ", file)
    time_and_space = file.split()
    time_and_space = list(set(time_and_space))

with open("Travel.txt", "rt", encoding = "utf-8") as file:
    file = file.read()
    file = re.sub(r"[\n]", " ", file)
    travel = file.split()
    travel = list(set(travel))

with open("Work and Business.txt", "rt", encoding = "utf-8") as file:
    file = file.read()
    file = re.sub(r"[\n]", " ", file)
    work_and_business = file.split()
    work_and_business = list(set(work_and_business))

# Opening the File with NLTK Stopwords 
with open("NLTK_stopwords.txt", "rt", encoding = "utf-8") as file:
    stop_words = file.read()
    stop_words = re.sub(r"[\n]", " ", stop_words)
    stop_words_in_the_text = stop_words.split()

# Reading the Text from the File for Analysis
with open("text_for_analysis.txt", "rt", encoding = "utf-8") as file:
    
    # Getting rid of tabulation symbols, URL adresses, mobile phones, e-mail names. Segmentation
    text = [line.strip().lower() for line in file]
    text_string = " ".join(text)
    text_without_punct = re.sub(r"[A-Za-z0-9\.]+@[a-z]{2,}\.[a-z]{2,}", "", text_string)
    text_without_punct = re.sub(r"8-[0-9]{3}-[0-9]{3}-[0-9]{2}-[0-9]{2}", "", text_without_punct)
    text_without_punct = re.sub(r"[0-9A-Za-zА-Яа-яёЁ]+.[a-z]{2,3}", "", text_without_punct)
    text_without_punct = re.sub(r"[^A-Za-z ]", "", text_string)
    
    # Tokenization and Lemmatization 
    document = nlp(text_without_punct)
    text_for_analysis = [token.lemma_ for token in document]
    text_by_words = nltk.word_tokenize(text_without_punct)

    # Getting rid of stopwords  
    text_for_analysis_final = [text_for_analysis[i] for i in range(len(text_for_analysis)) if text_for_analysis[i] not in stop_words]

    # Finding Named Entities in the Text 
    named_entities = Sentence(text_for_analysis_final)
    tagger.predict(named_entities)
    named_entities_in_the_text = [i for i in named_entities.get_spans("ner")]
        
# Classifying words according to the CEFR levels (from A1 to C1)
words_A1_in_the_text = np.array([text_for_analysis_final[i] for i in range(len(text_for_analysis_final)) for j in range(len(words_unique_A1_final)) if text_for_analysis_final[i] == words_unique_A1_final[j]])
words_A2_in_the_text = np.array([text_for_analysis_final[i] for i in range(len(text_for_analysis_final)) for j in range(len(words_unique_A2_final)) if text_for_analysis_final[i] == words_unique_A2_final[j]])
words_B1_in_the_text = np.array([text_for_analysis_final[i] for i in range(len(text_for_analysis_final)) for j in range(len(words_unique_B1_final)) if text_for_analysis_final[i] == words_unique_B1_final[j]])
words_B2_in_the_text = np.array([text_for_analysis_final[i] for i in range(len(text_for_analysis_final)) for j in range(len(words_unique_B2_final)) if text_for_analysis_final[i] == words_unique_B2_final[j]])
words_C1_in_the_text = np.array([text_for_analysis_final[i] for i in range(len(text_for_analysis_final)) for j in range(len(words_unique_C1_final)) if text_for_analysis_final[i] == words_unique_C1_final[j]])

# Counting a number of words by CEFR levels (from A1 to C1)
words_in_the_text = len(text_for_analysis_final) 
A1_number = len(words_A1_in_the_text)
A2_number = len(words_A2_in_the_text)
B1_number = len(words_B1_in_the_text)
B2_number = len(words_B2_in_the_text)
C1_number = len(words_C1_in_the_text)
overall_number = A1_number + A2_number + B1_number + B2_number + C1_number
NE_number = len(named_entities_in_the_text)
words_missing = words_in_the_text - overall_number - NE_number 

# Counting a percentage by CEFR levels (from A1 to C1)
percent_A1 = round(A1_number * 100 / words_in_the_text, 2)
percent_A2 = round(A2_number * 100 / words_in_the_text, 2)
percent_B1 = round(B1_number * 100 / words_in_the_text, 2)
percent_B2 = round(B2_number * 100 / words_in_the_text, 2)
percent_C1 = round(C1_number * 100 / words_in_the_text, 2)
percent_overall = percent_A1 + percent_A2 + percent_B1 + percent_B2 + percent_C1
percent_NE = round(NE_number * 100 / words_in_the_text, 2)
percent_missing = 100 - percent_overall - percent_NE

# Classifying collocations by the CEFR levels in the text
collocations_A1_in_the_text = np.array([collocations_A1_without_tags[i] for i in range(len(collocations_A1_without_tags)) if collocations_A1_without_tags[i] in text_without_punct])
collocations_A2_in_the_text = np.array([collocations_A2_without_tags[i] for i in range(len(collocations_A2_without_tags)) if collocations_A2_without_tags[i] in text_without_punct])
collocations_B1_in_the_text = np.array([collocations_B1_without_tags[i] for i in range(len(collocations_B1_without_tags)) if collocations_B1_without_tags[i] in text_without_punct])
collocations_B2_in_the_text = np.array([collocations_B2_without_tags[i] for i in range(len(collocations_B2_without_tags)) if collocations_B2_without_tags[i] in text_without_punct])
collocations_C1_in_the_text = np.array([collocations_C1_without_tags[i] for i in range(len(collocations_C1_without_tags)) if collocations_C1_without_tags[i] in text_without_punct])

# Counting a number of collocations by the CEFR levels in the text
collocations_by_levels = {"A1_collocations": len(collocations_A1_in_the_text), "A2_collocations": len(collocations_A2_in_the_text), "B1_collocations": len(collocations_B1_in_the_text), "B2_collocations": len(collocations_B2_in_the_text), "C1_collocations": len(collocations_C1_in_the_text)}
A1_number_collocations = len(collocations_A1_in_the_text)
A2_number_collocations = len(collocations_A2_in_the_text)
B1_number_collocations = len(collocations_B1_in_the_text)
B2_number_collocations = len(collocations_B2_in_the_text)
C1_number_collocations = len(collocations_C1_in_the_text)

# Classifying words according to the Oxford Dictionary's topics 
animals_in_the_text = np.array([animals[i] for i in range(len(animals)) if animals[i] in text_without_punct])
appearance_in_the_text = np.array([appearance[i] for i in range(len(appearance)) if appearance[i] in text_without_punct])
communication_in_the_text = np.array([communication[i] for i in range(len(communication)) if communication[i] in text_without_punct])
culture_in_the_text = np.array([culture[i] for i in range(len(culture)) if culture[i] in text_without_punct])
food_and_drink_in_the_text = np.array([food_and_drink[i] for i in range(len(food_and_drink)) if food_and_drink[i] in text_without_punct])
functions_in_the_text = np.array([functions[i] for i in range(len(functions)) if functions[i] in text_without_punct])
health_in_the_text = np.array([health[i] for i in range(len(health)) if health[i] in text_without_punct])
homes_and_buildings_in_the_text = np.array([homes_and_buildings[i] for i in range(len(homes_and_buildings)) if homes_and_buildings[i] in text_without_punct])
leisure_in_the_text = np.array([leisure[i] for i in range(len(leisure)) if leisure[i] in text_without_punct])
notions_in_the_text = np.array([notions[i] for i in range(len(notions)) if notions[i] in text_without_punct])
people_in_the_text = np.array([people[i] for i in range(len(people)) if people[i] in text_without_punct])
politics_and_society_in_the_text = np.array([politics_and_society[i] for i in range(len(politics_and_society)) if politics_and_society[i] in text_without_punct])
science_and_technology_in_the_text = np.array([science_and_technology[i] for i in range(len(science_and_technology)) if science_and_technology[i] in text_without_punct])
sports_in_the_text = np.array([sports[i] for i in range(len(sports)) if sports[i] in text_without_punct])
the_natural_world_in_the_text = np.array([the_natural_world[i] for i in range(len(the_natural_world)) if the_natural_world[i] in text_without_punct])
time_and_space_in_the_text = np.array([time_and_space[i] for i in range(len(time_and_space)) if time_and_space[i] in text_without_punct])
travel_in_the_text = np.array([travel[i] for i in range(len(travel)) if travel[i] in text_without_punct])
work_and_business_in_the_text = np.array([work_and_business[i] for i in range(len(work_and_business)) if work_and_business[i] in text_without_punct])

# Counting a number of words and collocations according to the Oxford Dictionary's topics 
animals_number = len(animals_in_the_text)
appearance_number = len(appearance_in_the_text)
communication_number = len(communication_in_the_text)
culture_number = len(culture_in_the_text)
food_and_drink_number = len(food_and_drink_in_the_text)
functions_number = len(functions_in_the_text)
health_number = len(health_in_the_text)
homes_and_buildings_number = len(homes_and_buildings_in_the_text)
leisure_number = len(leisure_in_the_text)
notions_number = len(notions_in_the_text)
people_number = len(people_in_the_text)
politics_and_society_number = len(politics_and_society_in_the_text)
science_and_technology_number = len(science_and_technology_in_the_text)
sports_number = len(sports_in_the_text)
the_natural_world_number = len(the_natural_world_in_the_text)
time_and_space_number = len(time_and_space_in_the_text)
travel_number = len(travel_in_the_text)
work_and_business_number = len(work_and_business_in_the_text)

# Creating a dictionary with words classified by topics
all_topical_numbers = {"Animals": animals_number, "Appearance": appearance_number, "Communication": communication_number, "Culture": culture_number, "Food and Drink": food_and_drink_number, "Functions": functions_number, "Health": health_number, "Homes and Buildings": homes_and_buildings_number, "Leisure": leisure_number, "Notions": notions_number, "People": people_number, "Politics and Society": politics_and_society_number, "Science and Technology": science_and_technology_number, "Sports": sports_number, "The Natural World": the_natural_world_number, "Time and Space": time_and_space_number, "Travel": travel_number, "Work and Business": work_and_business_number}
all_topical_numbers_sorted = sorted(all_topical_numbers.items(), key = lambda item: item[1], reverse = True)
topics = [all_topical_numbers_sorted[i][0] for i in range(len(all_topical_numbers_sorted))]
numbers = [all_topical_numbers_sorted[i][1] for i in range(len(all_topical_numbers_sorted))]

# Printing general statistics
general_statistics = pd.Series([len(nltk.sent_tokenize(text_string)), words_in_the_text], index=["A number of sentences in the text:", "A number of words (without stopwords) in the text:"] )
general_statistics.name = "General Statistics"
print(general_statistics)
print()

#Printing a number of words in the text classified by CEFR
words_by_CEFR = pd.Series([A1_number, A2_number, B1_number, B2_number, C1_number, overall_number], index=["A1 words in the text:", "A2 words in the text:", "B1 words in the text:", "B2 words in the text:", "C1 words in the text:", "Overall number of words classfied by CEFR in the text:"])
words_by_CEFR.name = "Words Classified by CEFR Levels (from A1 to C1)"
print(words_by_CEFR)
print()

# Printing a percentage of words in the text class classified by CEFR
percent_of_words = pd.Series([percent_A1, percent_A2, percent_B1, percent_B2, percent_C1, percent_overall], index=["Percent of the A1 words in the text:", "Percent of the A2 words in the text:", "Percent of the B1 words in the text:", "Percent of the B2 words in the text:", "Percent of the C1 words in the text:", "Overall percent of words (from A1 to C1) in the text (from 100%):"])
percent_of_words.name = "Percent of Words Classified by CEFR Levels (from A1 to C1) in the Text"
print(percent_of_words)
print()

# Printing a number and a percentage of Named Entities in the text
named_entities = pd.Series([NE_number, percent_NE], index=["A number of Named Entities in the text:", "A percent of Named Entities in the text (from 100%):"])
named_entities.name = "Named Entities in the Text"
print(named_entities)
print()

# Printing a number and a perentage of unindentified words in the text
unindentified_words = pd.Series([words_missing, percent_missing], index=["An overall number of unidentified words in the text:", "An overall percent of unidentified words in the text:"])
unindentified_words.name = "Unindentified Words in the Text"
print(unindentified_words)
print()

# Printing a number of collocations in the text classified by CEFR levels 
collocations = pd.Series([A1_number_collocations, A2_number_collocations, B1_number_collocations, B2_number_collocations, C1_number_collocations], index=["A1 collocations in the text:", "A2 collocations in the text:", "B1 collocations in the text:", "B2 collocations in the text:", "C1 collocations in the text:"])
collocations.name = "A Number of Collocations Classified by CEFR Levels (from A1 to C1) in the Text"
print(collocations)
print()

# Printing a number of words and collocations classified by the Oxford Dictionary's topics
all_topics = pd.Series(all_topical_numbers)
all_topics.name = "Words Classified by Topics"
print(all_topics.sort_values(ascending=False))
print()

popular_topics = pd.Series(np.arange(1, 6, 1), index=topics[:5])
popular_topics.name = "The Most Popular Topics"
print(popular_topics)

# Ending of the program
end = time.time() - start

# Dispaying program time
print()
print("Running time:", round(end, 2), "sec.")

# Results visualization. Words 
CEFR_levels = ["A1", "A2", "B1", "B2", "C1"]
words = [A1_number, A2_number, B1_number, B2_number, C1_number]

plt.bar(CEFR_levels, words)
plt.xlabel("CEFR levels", fontsize=20)
plt.ylabel("A Number of Words", fontsize=20)
plt.title("A Number of Words Classified by Levels (from A1 to C1) According to the CEFR", fontsize=25)
plt.tick_params(labelsize=16)
plt.show()

# Results visualization. Collocations
CEFR_levels = ["A1", "A2", "B1", "B2", "C1"]
collocations = [A1_number_collocations, A2_number_collocations, B1_number_collocations, B2_number_collocations, C1_number_collocations]

plt.bar(CEFR_levels, collocations)
plt.xlabel("CEFR Levels", fontsize=20)
plt.ylabel("A Number of Collocations", fontsize=20)
plt.title("A Number of Collocations in the Text Classified by CEFR", fontsize=25)
plt.tick_params(labelsize=16)
plt.show()

# Result visualization. Percent of words
x = [percent_A1, percent_A2, percent_B1, percent_B2, percent_C1, percent_NE, percent_missing]
levels = ["A1", "A2", "B1", "B2", "C1", "NE", "Missing"]
plt.pie(x)
plt.title("Percent of Words in the Text", fontsize=25)
plt.legend(title="Percent of Words in the Text", labels=levels, loc="upper left", bbox_to_anchor=(1.02, 1))
plt.show()

# Result Visualization. Topical words
plt.bar(topics[:5], numbers[:5])
plt.xlabel("Topics", fontsize=20)
plt.ylabel("A Number of Words", fontsize=20)
plt.title("The Most Popular Topics in the Text", fontsize=25)
plt.tick_params(labelsize=16)
plt.show()

