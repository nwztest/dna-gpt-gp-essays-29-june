import argparse
import os
import openai
from rouge_score.rouge_scorer import _create_ngrams
from nltk.stem.porter import PorterStemmer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, accuracy_score
import json
import re
import six
PorterStemmer = PorterStemmer()
import pprint

# parser: for python command line
# input_file is mandatory, output file is optional (default is output.html)
parser = argparse.ArgumentParser()
parser.add_argument('input_file')
parser.add_argument('-o', '--output_file', default='output.html')
parser.add_argument('-m', '--model', default='gpt-3.5-turbo')

# don't change this, this will cause the scores to be inaccurate, as the threshold set is hardcoded to be that of 10 regenerations and no golden prompt.
parser.add_argument('-k', '--number_of_generations', default=10, dest='k')

parser.add_argument('-t', '--truncate_ratio', default=0.5)
args = parser.parse_args()
args.k = int(args.k)

# ---------------------------------------------------
# step 1: essay text file read to essay
# ---------------------------------------------------
print("reading input file", args.input_file)
with open(os.path.join(os.getcwd(), args.input_file), "r", encoding="utf8") as input_file:
    essay = input_file.read()
print("essay loaded!\n")

# ---------------------------------------------------
# step 2: regeneration
# ---------------------------------------------------
print("regenerating", args.k, "time(s)")

# set-up
system_prompt = "You are an exemplary Singapore Junior College student that completes half-written essays from the sentences provided. You will write as many words as you can."
openai.api_key_path = os.path.join(os.getcwd(), "..", "..", "clingi_key")
# os.environ["REQUESTS_CA_BUNDLE"] = os.path.join(os.getcwd(), "..", "..", "certbundle.crt")
# os.environ["SSL_CERT_FILE"] = os.path.join(os.getcwd(), "..", "..", "certbundle.crt")

# truncate
essay_prefix = essay[:int(args.truncate_ratio * len(essay))]
essay_og_truncate = essay[ int( args.truncate_ratio*len(essay)): ]

# # regenerate
essay_regen_text = openai.ChatCompletion.create(
    model = args.model,
    messages = [
        {"role": "system", "content": system_prompt},
        # assume no gold prompt, results better without it
        {"role": "assistant", "content": essay_prefix}, # if there's gold prompt, it will appear here 
    ], 
    temperature = 1,
    max_tokens = 1500,
    n = args.k,
)

# improve formatting
essay_regen_list = []
for obj in essay_regen_text['choices']:
    essay_regen_list.append(obj['message']['content'])
    
print("regenerated!\n")

# ---------------------------------------------------
# step 3: n-gram analysis
# ---------------------------------------------------
print("analysing...")
# functions
def tokenize(text, stemmer, stopwords=[]):
    """Tokenize input text into a list of tokens.

    This approach aims to replicate the approach taken by Chin-Yew Lin in
    the original ROUGE implementation.

    Args:
    text: A text blob to tokenize.
    stemmer: An optional stemmer.

    Returns:
    A list of string tokens extracted from input text.
    """

    # Convert everything to lowercase.
    text = text.lower()
    # Replace any non-alpha-numeric characters with spaces.
    text = re.sub(r"[^a-z0-9]+", " ", six.ensure_str(text))

    tokens = re.split(r"\s+", text)
    if stemmer:
        # Only stem words more than 3 characters long.
        tokens = [stemmer.stem(x) if len(x) > 3 else x for x in tokens if x not in stopwords]

    # One final check to drop any empty or invalid tokens.
    tokens = [x for x in tokens if re.match(r"^[a-z0-9]+$", six.ensure_str(x))]

    return tokens

def get_score_ngrams(target_ngrams, prediction_ngrams):
    ngram_list = []
    intersection_ngrams_count = 0
    ngram_dict = {}
    for ngram in target_ngrams.keys(): # six.iterkeys(target_ngrams)
        num_of_intersections = min(target_ngrams[ngram], prediction_ngrams[ngram])
        intersection_ngrams_count += num_of_intersections
        ngram_dict[ngram] = num_of_intersections
        if num_of_intersections != 0 and (ngram not in ngram_list): ngram_list.append((ngram, num_of_intersections))
    target_ngrams_count = sum(target_ngrams.values()) # prediction_ngrams
    return intersection_ngrams_count / max(target_ngrams_count, 1), ngram_dict, ngram_list

def get_ngram_info(article_tokens, summary_tokens, _ngram):
    article_ngram = _create_ngrams( article_tokens , _ngram)
    summary_ngram = _create_ngrams( summary_tokens , _ngram)
    # print(article_ngram)
    ngram_score, ngram_dict, ngram_list = get_score_ngrams( article_ngram, summary_ngram)
    return ngram_score, ngram_dict, sum( ngram_dict.values() ), tuple(ngram_list)

def N_gram_detector(ngram_n_ratio):
    score = 0
    still_has_stuff = True
    non_zero = []
    gpt_ngrams = []
    
    for idx, key in enumerate(ngram_n_ratio):
        if idx in range(3) and 'score' in key or 'ratio' in key:
            score += 0. * ngram_n_ratio[ key ]
            continue
        if 'score' in key or 'ratio' in key:
            score += (idx+1) * np.log((idx+1))   * ngram_n_ratio[ key ]
            if ngram_n_ratio[ key ] != 0:
                non_zero.append( idx+1 )
        if 'list' in key:
            if ngram_n_ratio[key] != tuple():
                gpt_ngrams.append(ngram_n_ratio[key])
    
    # print(non_zero)
    # print(f"test score: {score}")
    return score / (sum( non_zero ) + 1e-8), gpt_ngrams # score / (sum( non_zero ) + 1e-8)

# analysis: ngram counts and discrete scores
essay_tokens = tokenize(essay_og_truncate, stemmer=PorterStemmer)
essay_analysis_dict_list = []
essay_analysis_dict = {}
for i in range(len(essay_regen_list)): # len(human_half)
    # print(i)
    essay_generate_tokens = tokenize(essay_regen_list[i], stemmer=PorterStemmer)
    if len(essay_generate_tokens) == 0:
        continue
    for _ngram in range(1, 25): # 4, 25
        ngram_score, ngram_dict, overlap_count, ngram_list = get_ngram_info(essay_tokens, essay_generate_tokens, _ngram)
        essay_analysis_dict['human_truncate_ngram_{}_score'.format(_ngram)] = ngram_score / len(essay_generate_tokens)
        essay_analysis_dict['human_truncate_ngram_{}_count'.format(_ngram)] = overlap_count
        essay_analysis_dict['human_truncate_ngram_{}_list'.format(_ngram)] = ngram_list
    essay_analysis_dict_list.append(essay_analysis_dict)

# print(json.dumps(essay_analysis_dict, indent=4))

# analysis: final score and AI likelihood
essay_ngrams_list_list = []
essay_final_score = 0
for essay_analysis_dict in essay_analysis_dict_list:
    essay_intermediate_score, essay_matching_ngrams = N_gram_detector(essay_analysis_dict)
    essay_final_score += essay_intermediate_score
    essay_ngrams_list_list.append(essay_matching_ngrams)
# print(essay_gpt_ngrams)
print("essay score:", essay_final_score, '\n')
if essay_final_score > 0.001375161528379888: result = "GENERATED BY AN AI"
else: result = "WRITTEN BY A HUMAN"



# analysis: ordered list of ngram list

essay_final_matching_ngrams = essay_matching_ngrams

# gave up :(
# for single_gen_ngrams in essay_ngrams_list_list:
#     for ngram_pair_tuple in single_gen_ngrams:
        

li_tags_list = []
# ordered_essay_matching_ngrams = []
for ngram_pair_tuple in essay_final_matching_ngrams:
    li_tags = ""
    ngram_pair_list = sorted(list(ngram_pair_tuple), key=lambda pair: pair[1], reverse=True)
    for ngram_pair in ngram_pair_list:
        li_tags += f"<li>{str(ngram_pair[0])}: {ngram_pair[1]}</li>"
    li_tags_list.append(li_tags)

# ---------------------------------------------------
# step 4: highlight insertion (ALPHA)
# ---------------------------------------------------
highlighted_texts = []
number_of_texts = len(essay_matching_ngrams)

# loops through each existing list of ngrams
for n in range(number_of_texts):
    ngram_list = essay_matching_ngrams[n]
    
    # splits string, does not split at \n
    # note: i made it such that it only looks at the second half now (29/6/2023)
    essay_split = re.findall(r'\S+|\n', essay_og_truncate)
    
    # warning: highly inefficient code!
    
    # loops through every ngram pair (meaning an angram and its count) of n in the tuple
    for ngram_pair in ngram_list:
        gram_idx = 0 # which word in the ngram you are at
        index_of_first = 0 # index of first ngram in the essay
        ngram_count = 0 # number of matching ngrams so far
        is_loop_broken = False
        
        # loops through each word in the essay
        for word_idx, word in enumerate(essay_split):
            
            if ngram_pair[0][gram_idx] in word.lower():
                if gram_idx == 0: index_of_first = word_idx
                gram_idx += 1
            else:
                gram_idx = 0
                continue
                
                
            if gram_idx == len(ngram_pair[0]):
                # writes <mark>
                essay_split[index_of_first] = "<mark>" + essay_split[index_of_first]
                
                # writes </mark>
                essay_split[word_idx] += "</mark>"
                
                # gram_idx back to 0
                gram_idx = 0
                
                # ngram_count increment by 1
                ngram_count += 1
                
                # exits this loop if all ngrams have been found.
                if ngram_count == ngram_pair[1]: 
                    is_loop_broken = True
                    break
                
        # if any discrepancy in count, user will be alerted
        if not is_loop_broken: print("discrepancy at ngram", ngram_pair[0])
        
    # rejoins essay
    essay_edited = ' '.join(essay_split)

    # to indicate it's the second half of the essay, an ellipsis is added
    essay_edited = "..." + essay_edited

    # essay cleanup: 
    # in case <br> appears in essay, we want to escape it.
    # newline is replaced with <br>
    essay_edited = essay_edited.replace('br>', 'br&gt;').replace('\n', '<br>')
    highlighted_texts.append(essay_edited)
    

# ---------------------------------------------------
# step 5: write output to html file
# ---------------------------------------------------
print("writing to output file", args.output_file)

html = f'''

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DNA-GPT Implementation</title>

    <!-- fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@300&family=Ubuntu+Mono:wght@400;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="styles.css">

</head>
<body>

    <div id="top">
        <h1 id="title">DNA-GPT Implementation</h1>
        <p class="subtitle">By Neo Wee Zen and Peh Yew Kee, Year 4, NUS High School, under attachment at DSO</p>
    </div>
    
    <div id="big-loud-disclaimer">
        <h1 id="result">YOUR ESSAY IS LIKELY <mark>{result}</mark></h1>
    </div>
    
    <div id="slider_div">
        <p id="ngram-p">bruh</p>
        <input type="range" min="1" max={number_of_texts} value={number_of_texts} class="slider" id="ngram-length-slider">
    </div>
    
    <div id="ngram-list_div">
        <h1 id="matched-h1">Matched n-grams</h1>
        <p class='subtitle'>The number on the right of each n-gram is the number of times the n-gram was found in the regenerated essays. Highlights shown are from the last essay generated.</p>
        <ol id='list'>
            {li_tags}
        </ol>
    </div>

    <div id="essay-output-div">
        <h1>(ALPHA) Highlighting of Essay</h1>
        <p id="essay-output">{essay_edited}</p>
    </div>
    
    <div id="nerd-stats">
    <h1>Nerd Stats</h1>
        <p>Model: {args.model}</p>
        <p>Number of regenerations: {args.k}</p>
        <p>Truncation ratio: {args.truncate_ratio}</p>
    </div>

    <script>

        text_arr = {str(highlighted_texts)}
        list_arr = {str(li_tags_list)}

        slider = document.getElementById('ngram-length-slider');
        nGramP = document.getElementById('ngram-p');
        matchedH1 = document.getElementById('matched-h1')
        
        nGramP.innerHTML = "n-gram length: " + slider.value;
        essayOutput = document.getElementById("essay-output")
        essayOutput.innerHTML = text_arr[slider.value - 1]
        list.innerHTML = list_arr[slider.value - 1]
        matchedH1.innerHTML = "Matched " + slider.value + "-grams"

        slider.oninput = function() {{
            nGramP.innerHTML = "n-gram length: " + slider.value
            essayOutput.innerHTML = text_arr[slider.value - 1]
            list.innerHTML = list_arr[slider.value - 1]
            matchedH1.innerHTML = "Matched " + slider.value + "-grams"
        }}

    </script>

</body>
</html>
'''
with open(args.output_file, 'w') as output_file:
    output_file.write(html)
print("written! now exiting programme...")