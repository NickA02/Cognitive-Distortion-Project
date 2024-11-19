import ollama
import pandas as pd
import re
import json
import sys

prompt = open('./prompts/researcher/researcher.txt', 'r').read()
prompt_pt2 = open('./prompts/researcher/researcherpt2.txt', 'r').read()


def request(prompt_text, patient_question, distortion, secondary_distortion, distorted_part):
    messages = []
    initial_prompt = {
        'role': 'system',
        'content': prompt_text
    }
    messages.append(initial_prompt)
    #print(f"System: {prompt_text}{patient_question}")
    messages = send_message(messages, patient_question)
    

    if distortion == 'No Distortion':
        second_step = "Your peer researchers in the field of Cognitive Behavioral Therapy have found that this patient was depicting an undistorted anecdote and that the patient's story is a reasonably genuine recount of their experience. "
        #print(f"System: {second_step}")

        messages = send_message(messages, second_step)
    else:
        if not secondary_distortion is None:
            second_step = f"Your peer researchers in the field of Cognitive Behavioral Therapy have found that this patient was depicting {distortion}. "
        else:
            second_step = f"Your peer researchers in the field of Cognitive Behavioral Therapy have found that this patient was depicting {distortion}, and possibly {secondary_distortion} as well. "
        second_step = second_step + f"The researchers support their claims through the following quote: '{distorted_part}'. "
        
    second_step = second_step + "Given these researchers are correct, please explain how they must've come to the conclusion and why you did not. Once you have thought this out, update your definitons of each cognitive distortion so that this patient's story can be better understood in the future."
    #print(f"System: {second_step}")
    messages = send_message(messages, second_step)
    return messages





def send_message(messages, chat):
    messages.append({
        'role': 'system',
        'content': chat
        })
    
    response = ollama.chat(
        model="gemma2:27b",
        messages=messages,
        stream=False
    )['message']['content']

    messages.append({
        'role': 'assistant',
        'content': response
    })
    #print(f"\n\nAssistant: {response}\n\n")
    return messages


def replace_definition(old_text, new_text):
    # Extract the new definition content between the tags
    new_definition_match = re.search(r'<START DEFINITION>(.*?)<END DEFINITION>', new_text, re.DOTALL)
    if new_definition_match:
        new_definition = new_definition_match.group(1)
    else:
        raise ValueError("New text does not contain a valid definition.")

    # Replace the old definition content with the new one
    updated_text = re.sub(r'<START DEFINITION>(.*?)<END DEFINITION>', f'<START DEFINITION>{new_definition}<END DEFINITION>', old_text, flags=re.DOTALL)

    return updated_text


sample = "Hola Hola <START DEFINITION> This is the old definition. <END DEFINITION>"
new_text = "Hello Hello <START DEFINITION> This is the new definition. <END DEFINITION>"

#print(replace_definition(sample, new_text))

chat_histories = []

sample_instances = pd.read_csv('datasets/train.csv')
j = 0
for sample_question in sample_instances.iterrows():
    sample_question = sample_question[1]
    sample_anecdote = sample_question['Patient Question']
    sample_distortion = sample_question['Dominant Distortion']
    sample_secondary_distortion = sample_question['Secondary Distortion (Optional)']
    sample_reason = sample_question['Distorted part']
    response = request(prompt + prompt_pt2, sample_anecdote, sample_distortion, sample_secondary_distortion, sample_reason)
    chat_histories.append(response)
    try:
        prompt = replace_definition(prompt, response[-1]['content'])
    except:
        print("\nFailed to redefine prompt")
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
    j = j + 1
    if j % 100 == 0:
        print(f"\niteration {j} definitions: {prompt}")
    sys.stdout.write(f"\r{j / len(sample_instances) * 100}% done")
    

# Save the chat histories to a file
with open('researcher_chat_histories.json', 'w') as f:
    json.dump(chat_histories, f, indent=4)

print(prompt)
