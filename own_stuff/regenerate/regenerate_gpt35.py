import argparse
import tqdm
import json
import os
import openai
import random
import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed
)

system_prompt = "You are an exemplary Singapore Junior College student that completes half-written essays from the sentences provided. You will write as many words as you can."

openai.api_key_path = os.getcwd() + "key path here"

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default=os.getcwd() + "/../../data/essay_gpt35_2.json")
parser.add_argument('--model', default= "gpt-3.5-turbo") # gpt-3.5-turbo, gpt-4-0314
parser.add_argument('--max_new_tokens', default=1500, type=int)
parser.add_argument('--regen_number', default=20, type=int)
parser.add_argument('--truncate_ratio', default=0.5, type=float)
parser.add_argument('--temperature', default= 1, type=float)
args = parser.parse_args()

# ai dataset file --> dictionary file

print('load model ...', args.model)
with open(args.dataset, "r", encoding='utf8') as gpt35_input_file:
    gpt35_dict = json.load(gpt35_input_file)
    
# human dataset file --> dictionary file

with open("../../data/essay_human_original.json", 'r', encoding='utf8') as human_file:
    human_dict = json.load(human_file)
    
# define output file

random.seed(43)
output_file = os.getcwd() + "/../../data/essay_regen_gpt35_" + str(args.truncate_ratio) + "_k20_gold.json"

# if the file exists it probably means that it was cut off halfway
# gpt35_dict is cut so as to pick up from where it left off
# already existing generations are skipped

if os.path.exists(output_file):
    with open(output_file, "r") as gpt35_input_file:
        output_json_temp = json.load(gpt35_input_file)
        if 'list' in output_json_temp.keys():
            print('valid json')
            num_curr_outputs = len(output_json_temp['list'])
            outputs = output_json_temp
        else: 
            print('different format')
            num_curr_outputs = 0
            outputs = {"list": []}
else:
    print('file doesn\'t exist')
    num_curr_outputs = 0
    outputs = {"list": []}

print("Skipping {} instances".format(num_curr_outputs))

input_json_count = 0
for website in gpt35_dict['a_level_gp']:
    for essay_obj in gpt35_dict['a_level_gp'][website]:
        input_json_count += 1

print('len(data): ', input_json_count)
# gpt35_dict = gpt35_dict[num_curr_outputs:]

# loop through dictionary (MODIFY)
# dd stands for defaultdict
@retry(wait=wait_fixed(15), stop=stop_after_attempt(60))
def completion_with_backoff():
    loop_count = 0
    for website_index, website in tqdm.tqdm(enumerate(gpt35_dict['a_level_gp']), total = len(gpt35_dict['a_level_gp'].keys())):
        for essay_index, essay_obj in tqdm.tqdm(enumerate(gpt35_dict['a_level_gp'][website]), total=len(gpt35_dict['a_level_gp'][website])):
            num_curr_outputs = len(outputs['list'])
            if loop_count >= num_curr_outputs:
                
                # human, machine are truncated accoriding to truncation ratio
                
                human_initial_text = human_dict['a_level_gp'][website][essay_index]['text']
                human_prefix_prompt = human_initial_text[ :int( args.truncate_ratio*len(human_initial_text) ) ]
                # print(human_initial_text[:50] + "...")

                ai_initial_text = essay_obj['response']
                machine_prefix_prompt = ai_initial_text[ :int( args.truncate_ratio*len(ai_initial_text) ) ]
                # print(ai_initial_text[:50] + "...")
                
                # human text regenerated [regen_number] times
                
                    
                human_gen_text = openai.ChatCompletion.create(model= args.model,
                            messages=[{"role": "system", "content": system_prompt},
                                    {"role": "user", "content": essay_obj['prompt']}, # mask out to simulate no golden prompt
                                        {
                                            "role": "assistant", 
                                            "content": human_prefix_prompt
                                        },
                                    ], 
                                    temperature= args.temperature,
                                    max_tokens = args.max_new_tokens,
                                    n= args.regen_number)
                
                time.sleep(10)
                
                # machine text regenerated [regen_number] times
                
                machine_gen_text = openai.ChatCompletion.create(model= args.model,
                            messages=[{"role": "system", "content": system_prompt},
                                    {"role": "user", "content": essay_obj['prompt']}, # mask out to simulate no golden prompt
                                        {
                                            "role": "assistant", 
                                            "content": machine_prefix_prompt
                                        },
                                    ], 
                                    temperature= args.temperature,
                                    max_tokens = args.max_new_tokens,
                                    n= args.regen_number)
                
                human_gen_text_list = []
                machine_gen_text_list = []
                
                for obj in human_gen_text['choices']:
                    human_gen_text_list.append(obj['message']['content'])
                for obj in machine_gen_text['choices']:
                    machine_gen_text_list.append(obj['message']['content'])
                
                
                
                
                outputs['list'].append({ 'prompt': essay_obj['prompt'],
                                            'human_gen_truncate': human_initial_text[ int( args.truncate_ratio*len(human_initial_text)): ],
                                            'machine_gen_truncate': ai_initial_text[ int( args.truncate_ratio*len(ai_initial_text) ): ], 
                                            "human_gen_text": human_gen_text_list, 
                                            "machine_gen_text": machine_gen_text_list })

                with open(output_file, "w") as gpt35_output_file:
                    json.dump(outputs, gpt35_output_file, indent=4)
                
            loop_count += 1
            # time.sleep(30)

completion_with_backoff()