import json
import re
import csv

def extract_definitions(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    definitions = []
    index = 1
    
    # Process each conversation thread
    for conversation in data:
        # Look for the last definition in each conversation
        pattern = r'<START DEFINITION>(.*?)<END DEFINITION>'
        last_definition = None
        
        for message in conversation:
            if 'content' in message:
                matches = re.findall(pattern, message['content'], re.DOTALL)
                if matches:
                    last_definition = matches[-1].strip()
        
        if last_definition:
            definitions.append([index, last_definition])
            index += 1
    
    # Write to CSV
    with open('definitions.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'definition'])  # header
        writer.writerows(definitions)

if __name__ == "__main__":
    extract_definitions('./researcher_chat_histories.json')