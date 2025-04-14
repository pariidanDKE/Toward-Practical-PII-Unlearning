import json
import numpy as np
from collections import defaultdict, Counter
from ahocorapy.keywordtree import KeywordTree
from timeit import default_timer as timer
import csv

# Code taken from https://github.com/albertsun1/gpt3-pii-attacks/blob/main/expt1.ipynb

# Function to extract and compare entities from fine-tuned and base model outputs
def test_set_difference(text_ft, text_base):
    entity_file = "ner_data_enron_wo_pretrained.json"
    k_value = None  # Controls if only unique entities are checked

    # Extract entity list by type
    def process_entity_type(entities, type):
        ent_list = []
        for k, v in entities.items():
            if k == type:
                for s in v:
                    entity = s.strip()
                    if len(entity) > 3:
                        ent_list.append(entity)
        return ent_list

    # Search for entity matches in the text
    def find_entities(entities, text, type, k_value=None):
        kwtree = KeywordTree(case_insensitive=False)
        val, cnt = np.unique(entities, return_counts=True)

        if k_value is not None:
            eidetic_ents = val[cnt == 1]
            for ent in eidetic_ents:
                kwtree.add(ent)
        else:
            unique_ents = set(val)
            for ent in unique_ents:
                kwtree.add(ent)

        kwtree.finalize()
        lines = text.split('\n')
        results = kwtree.search_all(' '.join(lines))
        result_set = set([result[0] for result in results])
        return result_set
    


    def main(entity_file):
        start = timer()

        all_count = 0

        all_ent = json.load(open("extractionfiles/" + entity_file))

        search_types = ["PERSON", "ORG", "LOC", "GPE", "FAC", "MONEY", "CARDINAL"]

        for type in search_types:
            select_ents = process_entity_type(all_ent, type)
            found_ents_1 = find_entities(select_ents, text_ft, type, k_value)
            found_ents_2 = find_entities(select_ents, text_base, type, k_value)
            set_diff = found_ents_1 - found_ents_2
            print(type, " count: ", len(set_diff))
            print(sorted(list(set_diff), key=len, reverse=True))
            all_count += len(set_diff)
            end = timer()
            # print("minute spent after ",type, ": ",(end-start)/60)
        print("Total number of entities found: ", all_count)

    main(entity_file)



# Alternate test function to extract all entities from a single model output
def test(concatenated_text):
    entity_file = "ner_data_enron_wo_pretrained.json"
    test_string = concatenated_text
    k_value = None

    def process_entity_type(entities, type):
        ent_list = []
        for k, v in entities.items():
            if k == type:
                for s in v:
                    entity = s.strip()
                    if len(entity) > 3:
                        ent_list.append(entity)
        return ent_list

    def find_entities(entities, text, type, k_value=None):
        kwtree = KeywordTree(case_insensitive=False)
        val, cnt = np.unique(entities, return_counts=True)

        if k_value is not None:
            eidetic_ents = val[cnt == 1]
            print("Number of ", k_value, " eidetic", type, " ents: ", eidetic_ents.size)
            for ent in eidetic_ents:
                kwtree.add(ent)
        else:
            unique_ents = set(val)
            print("Number of unique", type, " ents: ", len(list(unique_ents)))
            for ent in unique_ents:
                kwtree.add(ent)

        kwtree.finalize()
        lines = text.split('\n')
        results = kwtree.search_all(' '.join(lines))
        result_set = set([result[0] for result in results])
        return result_set
    

    def main(entity_file):
        start = timer()

        all_count = 0

        all_ent = json.load(open("extractionfiles/" + entity_file))

        search_types = ["PERSON", "ORG", "LOC", "GPE", "FAC", "MONEY", "CARDINAL"]

        for type in search_types:
            select_ents = process_entity_type(all_ent, type)
            found_ents = find_entities(select_ents, test_string, type, k_value)
            print(type, " count: ", len(found_ents))
            print(found_ents)
            all_count += len(found_ents)
            end = timer()
        print("Total number of entities found: ", all_count)
    
    main(entity_file)

def load_text_from_csv(file_path):
    """Helper function to load concatenated text from a CSV file."""
    concatenated_text = ''
    with open(file_path, mode='r') as file:
        print(f'Loading prompts from {file_path}...')
        reader = csv.reader(file, delimiter='|')
        next(reader)  # skip the header row
        for row in reader:
            concatenated_text += row[0].strip() + '\n'
    return concatenated_text

def main():
    concatenated_text_ft = load_text_from_csv('generatedfiles/full_test_autcomplete5%_ft.csv')
    concatenated_text_pt = load_text_from_csv('generatedfiles/full_test_autcomplete5%_pt.csv')

    #test(concatenated_text_ft)
    test_set_difference(text_ft=concatenated_text_ft,text_base=concatenated_text_pt)

if __name__ == "__main__":
    main()