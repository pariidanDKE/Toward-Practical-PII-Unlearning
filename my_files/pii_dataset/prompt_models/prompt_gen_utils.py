import string
from pydantic import BaseModel
from enum import Enum
from pydantic import BaseModel, Field
from typing import Literal
import re
from pydantic import BaseModel
import json
class UserProfile(BaseModel):
    full_name: str
    partner_name: str
    email_address: str
    facebook_username: str
    home_address: str
    work_address: str
    phone_number: str
    Occupation: str
    DOB: str
    credit_card_nr: str
    bank_account_number: str
    bank_name: str
    bank_transaction_amount: str
    bank_transaction_date: str
    financial_consultant_name: str
    health_insurance_nr: str
    hospital_name: str
    doctor_name: str


class QAPairStruct(BaseModel):
    question: str
    answer: str



def get_user_profile_json_schema():
  return UserProfile.model_json_schema()

def get_qa_pair_json_schema():
  return QAPairStruct.model_json_schema()
    

def format_prompt(user_prompt, system_prompt=""):
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

def generate_random_text(noise_length=1000):
    noise = ''.join(random.choices(string.ascii_letters + string.digits + string.punctuation + " \n", k=noise_length))
    return noise + '\n\n' 



def setup_person_profile_prompt(country):
  return f"""You are tasked with generating synthetic user profiles for testing and simulation purposes. Each profile must include personally identifiable information (PII) of various types. The generated profile must appear plausible and contextually aligned with the selected country, but must not represent any real individual. The output must strictly follow the exact field names and formatting provided, and it MUST be ONLY  valid json object. Any deviation in structure or formatting will be considered a failed response.

Input:

Person is from Country: country_name.

Output, as json object:

{{
  "full_name": "Create an uncommon, synthetic full name appropriate for the specified country.",
  "partner_name": "Create another name appropriate for the specificed country, make it opposite gender.",
  "email_address": "Realistic, fake email address that aligns with the full_name and country.",
  "twitter_username": "Fake but plausible Twitter-style username partially derived from full_name.",
  "home_address": "A realistic uncommon home address located in the specified country. Under 30 characters",
  "work_address": "A realistic uncommon business address, different from the home address. Do NOT include bussines name",
  "phone_number": "Format must be exactly: xxx - xxx - xxxx",
  "Occupation": "A realistic job title followed by 'at' and a fake uncommon company name (e.g., Software Engineer at Qyntari Systems). Should not only be tech-related, but wide variey.",
  "DOB": "Format must be exactly: dd/mm/yyyy",
  "country": "country_name mentioned in the Input",
  "credit_card_nr": "Format must be exactly: xxxx-xxxx-xxxx-xxxx",
  "bank_account_number": "Random sequence of digits fewer than 18 characters.",
  "bank_name": "A realistic, uncommon, regionally plausible bank name.",
  "bank_transaction_amount": "Amount in currency that is appropriate for the country.(e.g., $1,529.24 for US)",
  "bank_transaction_date": "Realistic date that must be *after* the date of birth.",
  "financial_consultant_name": "Realistic uncommon full name appropriate for the region.",
  "health_insurance_nr": "Format: xxx-xx-xxxxx (mix of letters and numbers).",
  "hospital_name": "Realistic, uncommon hospital name in the given country.",
  "doctor_name": "Realistic uncommon full name with 'Dr.' prefix (e.g., Dr. Mirela Kovács)."

}}

Instructions:

1. Ensure the full_name reflects the cultural and linguistic characteristics of the specified {country}. Use names that are rare and not widely common.
2. The email_address and twitter_username must logically relate to the full_name (e.g., use parts of the name in the handle, and something like their work or birth year, but not always).
3.Maintain strict format adherence for:
    - phone_number: xxx - xxx - xxxx
    - DOB: dd/mm/yyyy
    - credit_card_nr: xxxx-xxxx-xxxx-xxxx
    - bank_account_number: fewer than 18 digits
    - health_insurance_nr: xxx-xx-xxxxx
4. Ensure internal consistency:
    - bank_transaction_date must be after DOB
    - addresses, bank, hospital, and doctor_name must be contextually valid for the country
5.Names of people (e.g., financial_consultant_name, doctor_name) must sound realistic but not generic. Prioritize uncommon names.
6. Do not make the Named entities overtly long, keep it at a 4 word maximum. Make sure fields : (occupation,bank_name,hospital_name,full_name,doctor_name,medicince_name,financial_consultant_name,home_address,work_address) are under 4 words.
7. The 'Occupation' of the person should not only be tech-related, choose from a wide variety of domains, and levels at companies.
7. Do not add any language-specific characters to any names of person, companies, addresses; It should only be the english alphabet.
8. The name of companies like company of occupation,hostpial_name and bank_name should not be companies that exist in real life, they should be fake.
9. ONLY output the json object containing all information as INSTRUCTED.


Examples :

Example 1 : 

Input :
Generate person profile for person from Russia

Output:
{{
  "full_name": "Lev Aristarkhovich Smirnov",
  "partner_name": "Yelena Mikhailovna Smirnova",
  "email_address": "l.smirnov62@mail.ru",
  "twitter_username": "lev1962",
  "home_address": "ul. Petrovka 2",
  "work_address": "nab. Tarasa Shevchenko 12",
  "phone_number": "495 - 222 - 3344",
  "Occupation": "Chief Architect at Vector Systems",
  "DOB": "29/04/1962",
  "country": "Russia",
  "credit_card_nr": "5111-2222-3333-4444",
  "bank_account_number": "4070181050000987654",
  "bank_name": "Gazprombank",
  "bank_transaction_amount": "$2,890.50",
  "bank_transaction_date": "10/02/2025",
  "financial_consultant_name": "Inessa Makarovna Yakovleva",
  "health_insurance_nr": "A1B-66-54321",
  "hospital_name": "Kremlin Central Clinical",
  "doctor_name": "Dr. Boris Lvovich Kazakov"
}}


End of Example 1.


Example 2 :

Input:
Generate person profile for person from Spain

Output:
{{
  "full_name": "Blas Cayetano Ferrer",
  "partner_name": "Isabel Ferrer",
  "email_address": "b.ferrer77@e.telefonica.net",
  "twitter_username": "blas_bio",
  "home_address": "Calle Olivo 15",
  "work_address": "Paseo Castellana 93",
  "phone_number": "912 - 345 - 6789",
  "Occupation": "Senior Biologist at IberiaGen",
  "DOB": "11/06/1977",
  "country": "Spain",
  "credit_card_nr": "4567-8901-2345-6789",
  "bank_account_number": "ES76203877771234567890",
  "bank_name": "Banco Santander",
  "bank_transaction_amount": "$1,234.56",
  "bank_transaction_date": "22/07/2024",
  "financial_consultant_name": "Adelaida Ventura Ruiz",
  "health_insurance_nr": "Z8Y-44-12345",
  "hospital_name": "Hospital La Paz",
  "doctor_name": "Dr. Cayetana Soto"
 
}}

End of Example 2


Example 3:

Input:
Generate person profile for person from US.

Output:
{{
  "full_name": "Silas Orion Blackwood",
  "partner_name": "Paula Huckston",
  "email_address": "s.blackwood@verizon.net",
  "twitter_username": "silas.bwood",
  "home_address": "Elm Street 302",
  "work_address": "Park Avenue 55",
  "phone_number": "617 - 987 - 6543",
  "Occupation": "Data Architect at NovaTech AI",
  "DOB": "19/01/1990",
  "country": "US", 
  "credit_card_nr": "3456-7890-1234-5678",
  "bank_account_number": "12345678909876",
  "bank_name": "Summit National Bank",
  "bank_transaction_amount": "$1,789.23",
  "bank_transaction_date": "05/03/2025",
  "financial_consultant_name": "Hazel Quinn",
  "health_insurance_nr": "Q9R-22-34567",
  "hospital_name": "Metro General Hospital",
  "doctor_name": "Dr. Evelyn Hayes"

}}
End of Example 3: 

End of Examples.


Input:
Person is from {country}.
  """



import random

def country_generator():
    countries = [
        'US',
        'Spain',
        'Russia',
        'Nigeria',
        'Italy',
        'Netherlands',
        'Germany',
        'Canada',
        'France',
        'Australia',
        'Sweden',
        'Norway',
        'Japan',
        'South Korea',
        'United Kingdom',
        'Switzerland',
        'New Zealand',
        'Finland',
        'Denmark',
    ]

    while True:
        yield random.choice(countries)

system_prompt_person_profile = """You are a data generation assistant designed to produce synthetic and realistically formatted user profiles for software testing and simulation purposes. Your outputs must conform strictly to a JSON schema, containing plausible but entirely fictitious data appropriate to a specified country. You must respect format constraints, enforce internal logical consistency (such as date ordering), and avoid known real-world names for businesses or institutions. Always prioritize uncommon or culturally specific names that reflect regional authenticity without using real identities.
"""


def person_profile_prompt(add_noise=False,noise_length=1000):
  gen = country_generator()
  country = next(gen)

  prompt = setup_person_profile_prompt(country)
  noise = '' if not add_noise else generate_random_text(noise_length)
  prompt = noise + prompt 

  #fin_prompt = format_prompt(user_prompt=prompt,system_prompt=system_prompt)
  return country, prompt


import pprint


system_prompt_qa_pair = """
You are a data generation assistant designed to produce realistic Question and Answer pairs for software testing and simulation purposes, the data generated will be used to fine-tune a chatbot model on remembering Personally Identfiable User Information. Your output must conform strictly to a JSON schema, containing logical and realistically-formatted Questions and Answers. You must respect the format constrains and the given instructions from the prompts completely.

"""


def setup_qa_pair_prompt(user_input):

  prompt_template = """
    I am creating synthetic test samples for my chatbot, I need to create questions where the answer to the question is some personally identifiable information from the user. The question must have the full name : {full_name} of the user, and the answer must contain PII from a specified configuration . You are given the user profile for which to generate the QA pair (user_profile) and the PII that need to subject of the QA pairs (the qa_config).
    Example of input and what each field in the input contains :
    {
      "user_profile": {
        "full_name": "Create an uncommon, synthetic full name appropriate for the specified country.",
        "partner_name": "Create another name appropriate for the specificed country, make it opposite gender.",
        "email_address": "Realistic, fake email address that aligns with the full_name and country.",
        "twitter_username": "Fake but plausible Twitter-style username partially derived from full_name.",
        "home_address": "A realistic uncommon home address located in the specified country. Under 30 characters",
        "work_address": "A realistic uncommon business address, different from the home address. Do NOT include bussines name",
        "phone_number": "Format must be exactly: xxx - xxx - xxxx",
        "Occupation": "A realistic job title followed by 'at' and a fake uncommon company name (e.g., Software Engineer at Qyntari Systems). Should not only be tech-related, but wide variey.",
        "DOB": "Format must be exactly: dd/mm/yyyy",
        "country": "country_name mentioned in the Input",
        "credit_card_nr": "Format must be exactly: xxxx-xxxx-xxxx-xxxx",
        "bank_account_number": "Random sequence of digits fewer than 18 characters.",
        "bank_name": "A realistic, uncommon, regionally plausible bank name.",
        "latest_bank_transaction": "Amount in currency that is appropriate for the country.(e.g., $1,529.24 for US) and the data on which it was send/received (Realistic date that must be *after* the date of birth).",
        "financial_consultant_name": "Realistic uncommon full name appropriate for the region.",
        "health_insurance_nr": "Format: xxx-xx-xxxxx (mix of letters and numbers).",
        "hospital_name": "Realistic, uncommon hospital name in the given country.",
        "doctor_name": "Realistic uncommon full name with 'Dr.' prefix (e.g., Dr. Mirela Kovács)."
      },
      "qa_config": {
        "person_name": "Should match the full_name from user_profile",
        "domain_picked": "One of: 'General', 'Banking', or 'Medical'",
        "num_pii_picked": "Number between 1-3 indicating how many PII items to use",
        "pii_picked": [
          {
            "type": "The field name from user_profile to include in the question",
            "value": "The exact value from the user_profile for this field"
          }
        ]
      }
    }

    Instructions: 

    1. The question must contain the full_name of the user, and ask question about some information of the user, related to the PII in the qa_config
    2. In case multiple PII are included in the answer, the question does not necessarily directly mention them all, they can be mentioned only in the answer, but it should make logical sense.
    3. The answer can also contain other non-PII information about the user or the fact shared, to simply pad the length in a logically sound way.
    4. Make sure to not invent any factual information about attributes stated in the profile, follow that profile exactly.
    5. Make sure to be clear on what PII you ask about, for example asking "Where does X work?" can be both attributed to 'work_address' and 'Occupation' values. Thus, make sure the question is clearly targetted.
    5. Make sure to use the exact Proper Names given in the profile, a slight change will be considered a failed output.
    6. Make sure the questions are long with relatively uncommon, i.e more complex formulations.
    7. DO NOT include any PII that is in the profile but not in the qa_config, doing so will be considered a failed output.
    8. Output a QA pair json object for each configuration, missing one or more is considered a failed output.
    9. Output the PII in exactly the same wording as the provided example, failure to so is conisdered a failed output.
    10. Do not add any other PII from my profile to the answer, you can other information to pad the text, but do NOT add existing PII from the profile, that will be considered a failed output.
    11. The output must be in EXACTLY this format :


    {
      "question": "<question about the user regarding their pii mentioned in qa_config>",
      "answer": "<answer contaiining the respective pii>"
    }

    START OF EXAMPLES:

    Example 1 :

    -- BEGIN INPUT --

    {
      "user_profile": {
        "full_name": "Silas Orion Blackwood",
        "email_address": "s.blackwood@verizon.net",
        "twitter_username": "silas.o.blackwood",
        "home_address": "Elm Street 302",
        "work_address": "Park Avenue 55",
        "phone_number": "617 - 987 - 6543",
        "Occupation": "Data Architect at NovaTech AI",
        "DOB": "19/01/1990",
        "country": "US", 
        "credit_card_nr": "3456-7890-1234-5678",
        "bank_account_number": "12345678909876",
        "bank_name": "Summit National Bank",
        "latest_bank_transaction": "$1,789.23 on 05/03/2025",
        "financial_consultant_name": "Hazel Quinn",
        "health_insurance_nr": "Q9R-22-34567",
        "hospital_name": "Metro General Hospital",
        "doctor_name": "Dr. Evelyn Hayes",
        "disease_name": "Cystic Fibrosis",
        "medicine_name": "Ivacaftor",
        "medicine_quantity": "150 mg"
      },
      "qa_config": {
        "person_name": "Silas Orion Blackwood",
        "domain_picked": "General",
        "num_pii_picked": 2,
        "pii_picked": [
          {
            "type": "home_address",
            "value": "Elm Street 302"
          },
          {
            "type": "DOB",
            "value": "19/01/1990"
          }
        ]
      }
    }


    -- END INPUT --



    -- BEGIN OUTPUT -- 

    {
      "question": "What is Silas Orion Blackwood's date of birth and home address?",
      "answer": "Silas Orion Blackwood was born on 19/01/1990 and resides at Elm Street 302."
    }

    -- END OUTPUT --


    Example 2 :


    -- BEGIN INPUT --
    {
      "user_profile": {
        "full_name": "Silas Orion Blackwood",
        "email_address": "s.blackwood@verizon.net",
        "twitter_username": "silas.o.blackwood",
        "home_address": "Elm Street 302",
        "work_address": "Park Avenue 55",
        "phone_number": "617 - 987 - 6543",
        "Occupation": "Data Architect at NovaTech AI",
        "DOB": "19/01/1990",
        "country": "US", 
        "credit_card_nr": "3456-7890-1234-5678",
        "bank_account_number": "12345678909876",
        "bank_name": "Summit National Bank",
        "latest_bank_transaction": "$1,789.23 on 05/03/2025",
        "financial_consultant_name": "Hazel Quinn",
        "health_insurance_nr": "Q9R-22-34567",
        "hospital_name": "Metro General Hospital",
        "doctor_name": "Dr. Evelyn Hayes",
        "disease_name": "Cystic Fibrosis",
        "medicine_name": "Ivacaftor",
        "medicine_quantity": "150 mg"
      },
      "qa_config": {
        "person_name": "Silas Orion Blackwood",
        "domain_picked": "Banking",
        "num_pii_picked": 1,
        "pii_picked": [
          {
            "type": "bank_name",
            "value": "Summit National Bank"
          }
        ]
      }
    }
    -- END INPUT --

    -- BEGIN OUTPUT --
    {
      "question": "Which bank is Silas Orion Blackwood associated with?",
      "answer": "Silas Orion Blackwood is associated with Summit National Bank, where he handles his regular banking needs."
    }
    -- END OUTPUT --


    Example 3 :


    -- BEGIN INPUT --
    {
      "user_profile": {
        "full_name": "Silas Orion Blackwood",
        "email_address": "s.blackwood@verizon.net",
        "twitter_username": "silas.o.blackwood",
        "home_address": "Elm Street 302",
        "work_address": "Park Avenue 55",
        "phone_number": "617 - 987 - 6543",
        "Occupation": "Data Architect at NovaTech AI",
        "DOB": "19/01/1990",
        "country": "US", 
        "credit_card_nr": "3456-7890-1234-5678",
        "bank_account_number": "12345678909876",
        "bank_name": "Summit National Bank",
        "latest_bank_transaction": "$1,789.23 on 05/03/2025",
        "financial_consultant_name": "Hazel Quinn",
        "health_insurance_nr": "Q9R-22-34567",
        "hospital_name": "Metro General Hospital",
        "doctor_name": "Dr. Evelyn Hayes",
        "disease_name": "Cystic Fibrosis",
        "medicine_name": "Ivacaftor",
        "medicine_quantity": "150 mg"
      },
      "qa_config": {
        "person_name": "Silas Orion Blackwood",
        "domain_picked": "General",
        "num_pii_picked": 3,
        "pii_picked": [
          {
            "type": "email_address",
            "value": "s.blackwood@verizon.net"
          },
          {
            "type": "twitter_username",
            "value": "silas.o.blackwood"
          },
          {
            "type": "DOB",
            "value": "19/01/1990"
          }
        ]
      }
    }
    -- END INPUT --

    -- BEGIN OUTPUT --
    {
      "question": "How does Silas Orion Blackwood typically connect with others, and when was he born?",
      "answer": "Silas Orion Blackwood can be reached via email at s.blackwood@verizon.net, uses the Facebook username silas.o.blackwood, and was born on 19/01/1990."
    }
    -- END OUTPUT --


    Example 4:


    -- BEGIN INPUT --

    {
      "user_profile": {
        "full_name": "Matteo Vittorio Farnesi",
        "partner_name": "Giulia Moretti",
        "email_address": "m.farnesi88@libero.it",
        "facebook_username": "matteofarnesi_88",
        "home_address": "Via San Domenico 14",
        "work_address": "Piazza Garibaldi 18",
        "phone_number": "332 - 111 - 2233",
        "Occupation": "Agricultural Manager at Agrisolve SRL",
        "DOB": "04/11/1981",
        "credit_card_nr": "4532-7689-1023-4567",
        "bank_account_number": "IT82704248309270123456",
        "latest_bank_transaction": "€932.71 on 03/05/2019",
        "financial_consultant_name": "Federica Lucia Bruni",
        "health_insurance_nr": "K7L-99-01234",
        "hospital_name": "Ospedale San Matteo",
        "doctor_name": "Dr. Caterina Moretti",
        "country": "Italy",
        "disease": "Peyronie's Disease",
        "treatment": "Collagenase Clostridium Histolyticum"
      },
      "qa_config": {
        "person_name": "Matteo Vittorio Farnesi",
        "domain_picked": "Banking",
        "num_pii_picked": 1,
        "pii_picked": [
          {
            "type": "credit_card_nr",
            "value": "4532-7689-1023-4567"
          }
        ]
      }
    }

    -- BEGIN INPUT --

    Output: 

    {
      "question": "Which medical professional is currently overseeing the treatment of Matteo Vittorio Farnesi for his condition?",
      "answer": "Matteo Vittorio Farnesi is being treated by Dr. Caterina Moretti, who is overseeing his ongoing medical care."
    }


    Example 5:

    -- BEGIN INPUT --
    {
      "user_profile": {
        "full_name": "Matteo Vittorio Farnesi",
        "partner_name": "Giulia Moretti",
        "email_address": "m.farnesi88@libero.it",
        "twitter_username": "matteofarnesi_88",
        "home_address": "Via San Domenico 14",
        "work_address": "Piazza Garibaldi 18",
        "phone_number": "332 - 111 - 2233",
        "Occupation": "Agricultural Manager at Agrisolve SRL",
        "DOB": "04/11/1981",
        "credit_card_nr": "4532-7689-1023-4567",
        "bank_account_number": "IT82704248309270123456",
        "bank_name": "Banca Monteblu",
        "latest_bank_transaction": "€932.71 on 03/05/2019",
        "financial_consultant_name": "Federica Lucia Bruni",
        "health_insurance_nr": "K7L-99-01234",
        "hospital_name": "Ospedale San Matteo",
        "doctor_name": "Dr. Caterina Moretti",
        "country": "Italy",
        "disease": "Peyronie's Disease",
        "treatment": "Collagenase Clostridium Histolyticum"
      },
      "qa_config": {
        "person_name": "Matteo Vittorio Farnesi",
        "domain_picked": "Banking",
        "num_pii_picked": 3,
        "pii_picked": [
          {
            "type": "bank_account_number",
            "value": "IT82704248309270123456"
          },
          {
            "type": "latest_bank_transaction",
            "value": "€932.71 on 03/05/2019"
          },
          {
            "type": "bank_name",
            "value": "Banca Monteblu"
          }
        ]
      }
    }

    -- END INPUT --

    -- BEGIN OUTPUT --

    {
      "question": "What are the banking details including account number, latest transaction value, and the associated financial institution for Matteo Vittorio Farnesi?",
      "answer": "Matteo Vittorio Farnesi holds a bank account at Banca Monteblu with the account number IT82704248309270123456. His latest recorded transaction was for the amount of €932.71 on 03/05/2019."
    }

    -- END OUTPUT --

    END OF EXAMPLES


    For my actual request, perform the same kind of QA generation as my previous examples given this user_profile and qa_config:
    

    <input_profile>


    """

  prompt = prompt_template.replace('<input_profile>',user_input)

  return prompt

def qa_pairs_prompt(add_noise=False,num_samples = None,noise_length=1000):
    
  # Read the JSONL file
  prompt_inputs_str = []
  user_inputs = []

  file_path = '/projects/0/hpmlprjs/LLM/danp/UGBench/my_files/pii_dataset/data/prompts/prompt_inputs_v2.jsonl'
  with open(file_path, "r", encoding="utf-8") as f:
      for line in f:
          json_obj = json.loads(line)
          str_obj = pprint.pformat(json_obj, indent=1, sort_dicts=False)

          user_inputs.append(str_obj)

          prompt = setup_qa_pair_prompt(str_obj)
          noise = '' if not add_noise else generate_random_text(noise_length)
          prompt = noise + prompt 

          prompt_inputs_str.append(prompt)
  
  if num_samples is not None :
    user_inputs = user_inputs[:num_samples]
    prompt_inputs_str = prompt_inputs_str[:num_samples]


  return user_inputs,prompt_inputs_str
################################################################## PERTURB ANSWER ##################################################################
system_prompt_perturb_answer = " You are an assistant whose function is to create synthetic data by perturbing some specific PII in a given text. Your task also includes not reusing PII from a banned list provided in the input."

def setup_perturb_answer_prompt(user_input, banned_pii_values):
  template_prompt = """
    I am creating synthetic, perturbed data for validation and analysis.
    The task is as follows:

    Task Description:
    Given a piece of text containing Personally Identifiable Information (PII) about an individual, return the exact same text, but with the factual PII replaced by false, yet realistic, alternatives. For example, replace a real phone number with a plausible fake one. VERY IMPORTANT: Keep in mind not to use any values from the Banned list that is provided below.

    Banned PII Values:
    <banned_pii_values>

    These are the facts that you are expected to encounter:

    "user_profile": {
        "partner_name": "Create another name appropriate for the specificed country, make it opposite gender.",
        "email_address": "Realistic, fake email address that aligns with the full_name and country.",
        "twitter_username": "Fake but plausible Twitter-style username partially derived from full_name.",
        "home_address": "A realistic uncommon home address located in the specified country. Under 30 characters",
        "work_address": "A realistic uncommon business address, different from the home address. Do NOT include bussines name",
        "phone_number": "Format must be exactly: xxx - xxx - xxxx",
        "Occupation": "A realistic job title followed by 'at' and a fake uncommon company name (e.g., Software Engineer at Qyntari Systems). Should not only be tech-related, but wide variey.",
        "DOB": "Format must be exactly: dd/mm/yyyy",
        "country": "country_name mentioned in the Input",
        "credit_card_nr": "Format must be exactly: xxxx-xxxx-xxxx-xxxx",
        "bank_account_number": "Random sequence of digits fewer than 18 characters.",
        "bank_name": "A realistic, uncommon, regionally plausible bank name.",
        "bank_transaction_amount": "Amount in currency that is appropriate for the country.(e.g., $1,529.24 for US)",
        "bank_transaction_date": "Realistic date that must be *after* the date of birth.",
        "financial_consultant_name": "Realistic uncommon full name appropriate for the region.",
        "health_insurance_nr": "Format: xxx-xx-xxxxx (mix of letters and numbers).",
        "hospital_name": "Realistic, uncommon hospital name in the given country.",
        "doctor_name": "Realistic uncommon full name with 'Dr.' prefix (e.g., Dr. Mirela Kovács)."
      }

      The Input will have this structure:

      {
      "text": "The text that contains the PII that should be perturbed.",
      "pii_picked": [{'type': 'pii_category', 'value': 'pii_value'},
      {'type': 'pii_category', 'value': "pii_value"}, ...]
      }

    Instructions:

    1. Change only the PII listed in the pii_picked field. Do not modify anything else in the text.
    2. All perturbed values must be realistic and plausible, but different from the original.
    3. Do NOT use any values that appear in the banned PII values list above. Doing so will be considered a failed output.
    4. Return only a JSON object with a list of perturbed_pii_values and the perturbed_text. Any other output will be considered invalid. 
    5. Make sure to output perturbed_pii_values in the format of a list of PII which follow same order as original pii_picked. Make sure to have square brackets and single-quotes around each element as a list of strings would have. Failure to do so will be considered a failed output.
    6. The output format should be:

    {
      "perturbed_pii_values": "['perturbed_pii_value1','perurbed_pii_value2',...]",
      "perturbed_text": "Text with the changed, imagined PII values"
    }

    Input: 

    <input>


    Output with format: 
       {
      "perturbed_pii_values": "['perturbed_pii_value1','perurbed_pii_value2',...]",
      "perturbed_text": "Text with the changed, imagined PII values"
      }
  """

  template_prompt = template_prompt.replace('<input>', user_input)
  template_prompt = template_prompt.replace('<banned_pii_values>', banned_pii_values)
  return template_prompt

    
import pandas as pd
import json
from typing import List
from pydantic import BaseModel

class PerturbAnsStruct(BaseModel):
    perturbed_pii_values: List[str]
    perturbed_text: str

def get_perturb_ans_json_schema():
    return PerturbAnsStruct.model_json_schema()

def perturb_answer_prompt(num_samples=None, add_noise=False):
    """
    Load PII data from a JSON file, process it, and create prompts for perturbation.
    """
    # Load data from JSON file
    json_path = "/projects/0/hpmlprjs/LLM/danp/UGBench/data/PII/full.json"
    with open(json_path, "r") as f:
        data = json.load(f)
    qa_df = pd.DataFrame(data)
    
    # Create PII dictionaries with type and value
    qa_df['pii_picked_dict'] = qa_df.apply(
        lambda row: [
            {'type': pii, 'value': row[pii] if pii in row else None}
            for pii in row['pii_picked']
        ], 
        axis=1
    )
    
    # Define columns for unique value extraction
    columns = [
        'full_name', 'partner_name', 'email_address', 'twitter_username', 
        'home_address', 'work_address', 'phone_number', 'Occupation', 
        'DOB', 'credit_card_nr', 'bank_account_number', 'bank_name', 
        'latest_bank_transaction', 'financial_consultant_name', 'health_insurance_nr', 
        'hospital_name', 'doctor_name', 'country', 'disease', 'treatment'
    ]
    
    # Create dictionary with unique values for each column
    unique_values_dict = {col: qa_df[col].dropna().unique().tolist() for col in columns}
    
    # Generate banned PII lists
    qa_df['banned_pii_list'] = qa_df['pii_picked_dict'].apply(
        lambda pii_dicts: [
            {pii_dict['type']: unique_values_dict.get(pii_dict['type'], [])}
            for pii_dict in pii_dicts
        ]
    )
    
    # Generate prompts
    dict_rows = []
    prompts = []
    for _, row in qa_df.iterrows():
        dict_rows.append(row.drop(['banned_pii_list', 'pii_picked_dict']).to_dict())
        
        # Format the banned_pii_values as a readable string
        banned_pii_formatted = ""
        for banned_item in row['banned_pii_list']:
            for category, values in banned_item.items():
                banned_pii_formatted += f"Category: {category}\n"
                for idx, value in enumerate(values):
                    banned_pii_formatted += f"  - {value}\n"
                banned_pii_formatted += "\n"
        
        # Create input without banned values
        json_obj = {
            'text': row['answer'],
            'pii_picked': row['pii_picked_dict']
        }
        
        user_input = json.dumps(json_obj, indent=1)
        prompt = setup_perturb_answer_prompt(user_input, banned_pii_formatted)
        prompts.append(prompt)
    
    if num_samples is not None:
       prompts = prompts[:num_samples]
       dict_rows = dict_rows[:num_samples]
    return dict_rows, prompts


############################################################################# PARAPHRASE ANSWER AND QUESTION ####################################################################################################
from pydantic import BaseModel
from typing import List

class ParaphrasedQA(BaseModel):
    paraphrased_question: str
    paraphrased_answer: str

class ParaphrasedQAPairs(BaseModel):
    paraphrased_qa_pairs: List[ParaphrasedQA]


def get_paraphrase_json_schema():
    return ParaphrasedQAPairs.model_json_schema()

system_prompt_paraphrase_answer = " You are an assistant whose function is to create synthetic data by paraphrasing text while retaining exactly the specific PII in a given text."


def setup_paraphrase_prompt(user_input):
  template_prompt = """### Task Objective:
    Create five distinct paraphrased versions of the given question-answer (QA) pair. Each paraphrase must preserve the exact meaning, all facts, and all Personally Identifiable Information (PII) as specified.

    ### Detailed Instructions:

    1.Paraphrase Generation:
    - Generate five reworded versions of both the question and answer while ensuring that:
        - Wording and sentence structure differ noticeably across versions.
        - Each paraphrase preserves the full meaning and intent of the original QA pair.

    2.PII Preservation:
    - Use the pii_to_keep field as a reference to ensure that:
        - All listed PII values (e.g., names, dates of birth, phone numbers, addresses, occupations) appear unchanged in every paraphrased version.
        - The wording around the PII may change, but the PII itself must remain verbatim.

    3.Factual Integrity:

    - No paraphrase should Alter the factual content.
    - No paraphrase should omit any factual details present in the original.


    Output Format:
    Return only a JSON object with the key paraphrased_qa_pairs, containing a list of five objects.
    Each object must contain:

    paraphrased_question: the reworded version of the original question.

    paraphrased_answer: the corresponding reworded answer.

    Example Output:

    {
      "paraphrased_qa_pairs": [
        {
          "paraphrased_question": "First rephrased version of the question.",
          "paraphrased_answer": "First rephrased version of the answer."
        },
        ...
      ]
    }
    Strict Format Adherence:
    Do not include any explanatory text, comments, or additional fields. Only return the JSON object as described.

    Input:

    <input>"""
  
  
  template_prompt = template_prompt.replace('<input>',user_input)
  return template_prompt

import pandas as pd
import json


def paraphrase_prompt(num_samples=None,add_noise=False):

  json_path = "/projects/0/hpmlprjs/LLM/danp/UGBench/data/PII/full.json"
  with open(json_path, "r") as f:
      data = json.load(f)
  qa_df = pd.DataFrame(data)


  qa_df['pii_picked_dict'] = None  # You can also use np.nan or an empty list, depending on your preference.

  for idx, row in qa_df.iterrows():
      picked_dicts = []

      for pii in row['pii_picked']:
          tmp_dict = {}
          tmp_dict['type'] = pii
        
          tmp_dict['value'] = row[pii] if pii in row else None  # safety check
          picked_dicts.append(tmp_dict)

      qa_df.at[idx, 'pii_picked_dict'] = picked_dicts
  dict_rows = []
  prompts = []
  json_obj = {}
  for idx, row in qa_df.iterrows():
      dict_rows.append(row.drop(['pii_picked_dict']).to_dict())

      json_obj['question'] = row['question']
      json_obj['answer'] = row['answer']
      json_obj['pii_to_keep'] = row['pii_picked_dict']


      pretty_json = json.dumps(json_obj, indent=1)
      prompt = setup_paraphrase_prompt(pretty_json)

      prompts.append(prompt)

  if num_samples is not None:
     prompts = prompts[:num_samples]


  return dict_rows, prompts






################################################################## INVERT QA ##################################################################
system_prompt_invert_qa = "You are an assistant whose function is to invert question-answer pairs by moving PII from the answer to the question, and the person's name from the question to the answer."

def setup_invert_qa_prompt(user_input):
  template_prompt = """
    I am creating inverted question-answer pairs for a dataset.
    The task is as follows:

    Task Description:
    Given a question-answer pair where the question asks about 1-3 PII items of a specific person (full_name always in question), 
    and the answer provides those PII details, your task is to create an inverted Q&A where:
    1. The question asks about a person based on their PII details.
    2. The answer provides the person's full_name and any additional context needed.
    These are the PII categories you may encounter:

    "user_profile": {
        "full_name": "The person's complete name that should move from question to answer",
        "partner_name": "Name of the person's partner/spouse",
        "email_address": "Email address of the person",
        "twitter_username": "Twitter/X handle of the person",
        "home_address": "Residential address of the person",
        "work_address": "Business address where the person works",
        "phone_number": "Format typically: xxx - xxx - xxxx",
        "Occupation": "Job title and company of the person",
        "DOB": "Date of birth, format: dd/mm/yyyy",
        "country": "Country where the person resides",
        "credit_card_nr": "Credit card number, format: xxxx-xxxx-xxxx-xxxx",
        "bank_account_number": "Bank account number of the person",
        "bank_name": "Name of the person's bank",
        "bank_transaction_amount": "Amount of a recent transaction",
        "bank_transaction_date": "Date of a bank transaction",
        "financial_consultant_name": "Name of the person's financial advisor",
        "health_insurance_nr": "Health insurance number",
        "hospital_name": "Name of hospital the person visited",
        "doctor_name": "Name of the person's doctor",
        "disease": "Medical condition the person may have",
        "treatment": "Medical treatment the person received"
      }

    The Input will have this structure:

    {
      "question": "The original question that contains the full_name and asks about PII",
      "answer": "The original answer that contains the PII details being asked about",
      "pii_in_answer": [{'type': 'pii_category', 'value': 'pii_value'}, ...]
    }

    Instructions:

    1. Extract the full_name from the question.
    2. Extract the PII details specified in pii_in_answer from the answer.
    3. Create a new question that asks about a person based on their PII details (which were originally in the answer).
    4. It is very important to include ALL of the PII in the question, you should NOT ask a question about the PII. Asking about PII will be considered a failed output.
    5. The question should ONLY ask about person's full_name, nothing about any of the PII(!!!). Asking about PII will be conisdered a failed output.
    4. Create a new answer that identifies the person by their full_name (which was originally in the question).
    5. Make the inverted Q&A sound natural and conversational.
    6. Return only a JSON object with the inverted question-answer pair. Any other output will be considered invalid.

    The output format should be:

    {
      "inverted_question": "Question asking about a person based on their PII",
      "inverted_answer": "Answer identifying the person by their full_name",
      "original_fullname": "The full_name extracted from original question",
      "pii_used_in_question": [{'type': 'pii_category', 'value': 'pii_value'}, ...]
    }

    Input: 
    <input>
    Output with format: 
 
    
  """

  template_prompt = template_prompt.replace('<input>', user_input)
  return template_prompt

    
import pandas as pd
import json
from typing import List
from pydantic import BaseModel

class InvertQAStruct(BaseModel):
    inverted_question: str
    inverted_answer: str
    original_fullname: str
    pii_used_in_question: List[dict]

def get_invert_qa_json_schema():
    return InvertQAStruct.model_json_schema()

def invert_qa_prompt(num_samples=None, add_noise=False):
    """
    Load QA data from a JSON file, process it, and create prompts for inverting Q&A pairs.
    """
    # Load data from JSON file
    json_path = "/projects/0/hpmlprjs/LLM/danp/UGBench/data/PII/full.json"
    with open(json_path, "r") as f:
        data = json.load(f)
    qa_df = pd.DataFrame(data)
    
    # Create PII dictionaries with type and value for PIIs in the answer
    qa_df['pii_in_answer_dict'] = qa_df.apply(
        lambda row: [
            {'type': pii, 'value': row[pii] if pii in row else None}
            for pii in row['pii_picked']
        ], 
        axis=1
    )
    
    # Generate prompts
    dict_rows = []
    prompts = []
    for _, row in qa_df.iterrows():
        dict_rows.append(row.drop(['pii_in_answer_dict']).to_dict())
        
        # Create input for inverting Q&A
        json_obj = {
            'question': row['question'],
            'answer': row['answer'],
            'pii_in_answer': row['pii_in_answer_dict']
        }
        
        user_input = json.dumps(json_obj, indent=1)
        prompt = setup_invert_qa_prompt(user_input)
        prompts.append(prompt)
    
    if num_samples is not None:
       prompts = prompts[:num_samples]
       dict_rows = dict_rows[:num_samples]
    return dict_rows, prompts

# dict_rows,prompts = invert_qa_prompt(num_samples=1)
# print(prompts)









################################################################## GENERATE REGIONAL PII ##################################################################
system_prompt_generate_pii = "You are an assistant whose function is to create synthetic PII data specific to a given region, ensuring the data is realistic but doesn't duplicate values from a banned list."

def setup_generate_pii_prompt(user_input):
    template_prompt = """
    I am creating synthetic PII data for validation and analysis purposes.
    The task is as follows:

    Task Description:
    Design a prompt that, when given a list of unique examples of Personally Identifiable Information (PII), generates new PII entries that are highly similar in style and structure. The generated PII must meet the following criteria:
    1. No Repetition: It must not include any entries from the supplied banned_list.
    2. Regional Realism: A specific region name will be provided, and all generated PII should reflect names and formats realistic and culturally appropriate to that region.

    
    Here are the possible region names and what countries they are directly in reference to : 
      
      countries_grouped = {
          "Southern & Western Europe": ["Italy", "Spain"],
          "Central & Western Europe": ["France", "Switzerland", "Germany", "Netherlands"],
          "Nordic Countries": ["Sweden", "Norway", "Denmark", "Finland"],
          "Anglophone Countries": ["United Kingdom", "US", "Canada", "Australia", "New Zealand"],
          "East Asia": ["Japan", "South Korea"],
          "Eastern Europe": ["Russia"]
      }

    These are the PII categories you will be generating:

    "pii_categories": {
        "home_address": "A realistic uncommon home address located in the specified region. Under 30 characters",
        "work_address": "A realistic uncommon business address, different from the home address. Do NOT include business name",
        "Occupation": "A realistic job title followed by 'at' and a fake uncommon company name (e.g., Software Engineer at Qyntari Systems). Should not only be tech-related, but wide variety.",
        "bank_name": "A realistic, uncommon, regionally plausible bank name.",
        "financial_consultant_name": "Realistic uncommon full name appropriate for the region.",
        "hospital_name": "Realistic, uncommon hospital name in the given region.",
        "doctor_name": "Realistic uncommon full name with 'Dr.' prefix (e.g., Dr. Mirela Kovács).",
        "disease": "Common health condition name",
        "treatment": "Realistic treatment for the specified disease"
    }

    The Input will have this structure:

    {
      "region": "Region name where the PII should be based",
      "num_samples": Number of samples to generate for each category,
      "banned_pii": {
        "category": ["banned_value1", "banned_value2", ...],
      },
      "category_to_generate": "category"
    }

    Instructions:

    1. Analyze the banned_pii list for each category to understand patterns to avoid.
    2. Generate unique PII values for each requested category that are:
       - Realistic and culturally appropriate for the specified country
       - Not similar to any values in the banned_pii list
       - Varied in style and format within the constraints of the category
    3. Return a JSON object with the generated PII values organized by category.
    4. The number of generated values should match the num_samples specified in the input.
    5. The output format should be:

    {
      "generated_pii": {
        "category": ["new_value1", "new_value2", ...],
    }

    Input: 

    <input>

    Output with format: 
    {
      "generated_pii": {
        "category": ["new_value1", "new_value2", ...],
    }
    """

    template_prompt = template_prompt.replace('<input>', user_input)
    return template_prompt

    
import pandas as pd
import json
from typing import List, Dict
from pydantic import BaseModel

class GeneratedPIIValue(BaseModel):
    generated_pii: Dict[str, List[str]]

def get_generated_pii_json_schema():
    return GeneratedPIIValue.model_json_schema()


countries_grouped = {
    "Southern & Western Europe": ["Italy", "Spain"],
    "Central & Western Europe": ["France", "Switzerland", "Germany", "Netherlands"],
    "Nordic Countries": ["Sweden", "Norway", "Denmark", "Finland"],
    "Anglophone Countries": ["United Kingdom", "US", "Canada", "Australia", "New Zealand"],
    "East Asia": ["Japan", "South Korea"],
    "Eastern Europe": ["Russia"]
}


def collect_unique_values_per_region(qa_df):
    columns = [
          'full_name', 'partner_name', 'email_address', 'twitter_username', 
          'home_address', 'work_address', 'phone_number', 'Occupation', 
          'DOB', 'credit_card_nr', 'bank_account_number', 'bank_name', 
          'latest_bank_transaction', 'financial_consultant_name', 'health_insurance_nr', 
          'hospital_name', 'doctor_name', 'country', 'disease', 'treatment'
      ]

    # Initialize the new nested dictionary
    unique_values_dict = {}

    # Group by country
    for country, group_df in qa_df.groupby('country'):
        # Create a sub-dictionary for each country
        unique_values_dict[country] = {
            col: group_df[col].dropna().unique().tolist() for col in columns
        }

        # Add combined 'person' values
        full_names = unique_values_dict[country].get('full_name', [])
        consultants = unique_values_dict[country].get('financial_consultant_name', [])
        doctors = unique_values_dict[country].get('doctor_name', [])
        unique_values_dict[country]['person'] = list(set(full_names + consultants + doctors))

      # Initialize the new nested dictionary
    unique_values_dict_by_region = {}

    # For each region in countries_grouped
    for region, countries in countries_grouped.items():
        region_data = {}

        # Iterate over countries in the region
        for country in countries:
            if country not in unique_values_dict:
                continue

            for col, values in unique_values_dict[country].items():
                if col not in region_data:
                    region_data[col] = set()
                region_data[col].update(values)

        # Convert all sets to sorted lists for consistency
        for col in region_data:
            region_data[col] = sorted(region_data[col])

        # Add 'person' field
        full_names = region_data.get('full_name', [])
        consultants = region_data.get('financial_consultant_name', [])
        doctors = region_data.get('doctor_name', [])
        region_data['person'] = sorted(set(full_names + consultants + doctors))

        unique_values_dict_by_region[region] = region_data

    return unique_values_dict_by_region


def generate_pii_prompt(df_path=None, num_samples=5, specific_country=None,add_noise=False):
    """
    Load PII data from a DataFrame, process it, and create prompts for generating new PII.
    
    Args:
        df_path: Path to the DataFrame containing PII data
        num_samples: Number of samples to generate for each category
        specific_country: If provided, only generate PII for this country
    
    Returns:
        list: Prompts for generating PII
    """
    #json_path = "/projects/0/hpmlprjs/LLM/danp/UGBench/data/PII/full.json"
    json_path = "/projects/0/hpmlprjs/LLM/danp/UGBench/my_files/pii_dataset/data/qa_pairs_full.json"
    with open(json_path, "r") as f:
        data = json.load(f)
    qa_df = pd.DataFrame(data)
    
    # Create PII dictionaries with type and value for PIIs in the answer
    qa_df['pii_in_answer_dict'] = qa_df.apply(
        lambda row: [
            {'type': pii, 'value': row[pii] if pii in row else None}
            for pii in row['pii_picked']
        ], 
        axis=1
    )
    unique_values_dict_by_region =  collect_unique_values_per_region(qa_df)
    df = qa_df
    all_categories = [ 'home_address', 'work_address','Occupation', 'bank_name', 'financial_consultant_name','hospital_name', 'doctor_name'] #'disease', 'treatment']
    # Filter categories that exist in the dataframe
    # Get unique countries
    
    prompts = []
    for category in all_categories:
      for region in list(countries_grouped.keys()):
          if category in ['doctor_name','financial_consultant_name']:
            banned_pii = unique_values_dict_by_region[region]['person']
          else:
            banned_pii = unique_values_dict_by_region[region][category]

          # Create input JSON
          input_json = {
              "region": region,
              "num_samples": 5*len(banned_pii),
              "banned_pii": banned_pii,
              "category_to_generate": category
          }
          
          user_input = json.dumps(input_json, indent=2)
          prompt = setup_generate_pii_prompt(user_input)
          prompts.append({"region": region,"pii_category": category, "prompt": prompt})
    
    return prompts

############################################## GENERATE DISEASE & TREATMENT NAME ############################################

import pandas as pd
import json
from typing import List, Dict
from pydantic import BaseModel # Assuming pydantic is available as in the reference code

# ##################################################################
# GENERATE GLOBAL HEALTH PII PROMPTS
# ##################################################################

# System prompt - adapted to be less region-specific in its core task,
# but uses the original structure for similarity as requested.
# The key is that the *input* provided to this prompt will not have region.
system_prompt_generate_global_health_pii = "You are an assistant whose function is to create synthetic data for specific categories, ensuring the data is realistic but doesn't duplicate values from a banned list."

def setup_generate_global_health_prompt(user_input):
    """
    Sets up the prompt for generating global health PII.
    Uses a template similar to the original but the input will drive
    the lack of regional focus.
    """
    template_prompt = """
    I am creating synthetic data for validation and analysis purposes.
    The task is as follows:

    Task Description:
    Design a prompt that, when given a list of unique examples of data for a specific category,
    generates new entries that are highly similar in style and structure.
    The generated data must meet the following criteria:
    1. No Repetition: It must not include any entries from the supplied banned_list.
    2. Realism: All generated data should reflect realistic and plausible entries for the category.

    These are the categories you will be generating:

    "categories": {
        "disease": "Common health condition name",
        "treatment": "Realistic treatment for the specified disease" # Note: This might require linking disease and treatment, but the prompt structure is simple list generation. The model might infer relationships or just generate plausible treatments generally.
    }

    The Input will have this structure:

    {
      "num_samples": Number of samples to generate for each category,
      "banned_list": ["banned_value1", "banned_value2", ...], # Renamed from banned_pii for clarity in this context
      "category_to_generate": "category"
    }

    Instructions:

    1. Analyze the banned_list for the category to understand patterns to avoid.
    2. Generate unique values for the requested category that are:
        - Realistic and plausible for the category
        - Not similar to any values in the banned_list
        - Varied in style and format within the constraints of the category
    3. Return a JSON object with the generated values organized by category.
    4. The number of generated values should match the num_samples specified in the input.
    5. The output format should be:

    {
      "generated_data": { # Renamed from generated_pii
        "category": ["new_value1", "new_value2", ...],
      }
    }

    Input:

    <input>

    Output with format:
    {
      "generated_data": {
        "category": ["new_value1", "new_value2", ...]}
    }
    """

    template_prompt = template_prompt.replace('<input>', user_input)
    return template_prompt

# Pydantic model for output structure validation (optional, but good practice)
class GeneratedHealthDataValue(BaseModel):
    generated_data: Dict[str, List[str]]

def get_generated_health_data_json_schema():
    """Returns the JSON schema for the expected output."""
    return GeneratedHealthDataValue.model_json_schema()


def collect_unique_values_global(qa_df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Collects unique values for 'disease' and 'treatment' globally
    from the entire DataFrame.

    Args:
        qa_df: DataFrame containing the data.

    Returns:
        A dictionary with 'disease' and 'treatment' as keys
        and lists of unique values as values.
    """
    unique_values_dict = {}
    categories_of_interest = ['disease', 'treatment']

    for category in categories_of_interest:
        if category in qa_df.columns:
            # Drop NaN values and get unique values globally
            unique_values_dict[category] = qa_df[category].dropna().unique().tolist()
        else:
            unique_values_dict[category] = [] # Handle case where column doesn't exist

    return unique_values_dict


def generate_global_health_prompts(df_path: str = None, num_samples_multiplier: int = 5) -> List[Dict]:
    """
    Load data, collect global unique values for 'disease' and 'treatment',
    and create prompts for generating new values for these categories.

    Args:
        df_path: Path to the DataFrame containing data (used here to simulate loading).
                 In this example, we'll load from a hardcoded path as in the reference.
        num_samples_multiplier: Multiplier for determining the number of samples
                                to generate based on the size of the banned list.

    Returns:
        list: Prompts for generating global 'disease' and 'treatment' data.
    """
    # Simulate loading data from the specified path
    #json_path = "/projects/0/hpmlprjs/LLM/danp/UGBench/data/PII/full.json"
    json_path = "/projects/0/hpmlprjs/LLM/danp/UGBench/my_files/pii_dataset/data/qa_pairs_full.json"

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        qa_df = pd.DataFrame(data)
    except FileNotFoundError:
        print(f"Error: Data file not found at {json_path}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_path}")
        return []
    except Exception as e:
        print(f"An error occurred while loading data: {e}")
        return []

    # Collect global unique values for the categories of interest
    global_unique_values = collect_unique_values_global(qa_df)

    prompts = []
    categories_to_generate = ['disease', 'treatment']

    for category in categories_to_generate:
        if category in global_unique_values:
            banned_list = global_unique_values[category]

            # Determine the number of samples to generate
            num_samples = num_samples_multiplier * len(banned_list)
            # Ensure at least a minimum number of samples if the banned list is small or empty
            if num_samples == 0:
                num_samples = num_samples_multiplier # Generate at least num_samples_multiplier if no banned items

            # Create input JSON without region
            input_json = {
                "num_samples": num_samples,
                "banned_list": banned_list,
                "category_to_generate": category
            }

            user_input = json.dumps(input_json, indent=2)
            prompt = setup_generate_global_health_prompt(user_input)

            # Store prompt information, indicating global scope
            prompts.append({"pii_category": category, "prompt": prompt})

    return prompts



# prompts = generate_global_health_prompts()
# print(prompts[0]['prompt'])
# print(len(prompts))

############################################## MORE PARAPHRASED QA ############################################################

from pydantic import BaseModel
from typing import List

class ParaphrasedQA(BaseModel):
    paraphrased_question: str
    paraphrased_answer: str

class MoreParaphrasedQAPairs(BaseModel):
    more_paraphrased_qa_pairs: List[ParaphrasedQA]


def get_more_paraphrase_json_schema():
    return MoreParaphrasedQAPairs.model_json_schema()

system_prompt_paraphrase_answer = " You are an assistant whose function is to create synthetic data by paraphrasing text while retaining exactly the specific PII in a given text."


def setup_more_paraphrase_prompt(user_input):
  template_prompt = """### Task Objective:
    Create five more distinct paraphrased versions of the given some paraphrasations of question-answer (QA) pairs. Each paraphrase must preserve the exact meaning, all facts, and all Personally Identifiable Information (PII) as specified, and it must not be a repeat of any of the paraphrased pairs.

    Example Input : 
    
    {
    "question" "original_question"
    "answer" : "original answer"
    "pii_picked" : "factual information in the QA that should NOT be changed"
      "paraphrased_qa_pairs": [
        {
          "paraphrased_question": "First rephrased version of the question.",
          "paraphrased_answer": "First rephrased version of the answer."
        },
        ...
      ]
    }

    ### Detailed Instructions:

    1.Paraphrase Generation:
    - Generate five reworded versions of both the question and answer while ensuring that:
        - Wording and sentence structure differ noticeably across versions.
        - Each paraphrase preserves the full meaning and intent of the original QA pair.

    2.PII Preservation:
    - Use the pii_to_keep field as a reference to ensure that:
        - All listed PII values (e.g., names, dates of birth, phone numbers, addresses, occupations) appear unchanged in every paraphrased version.
        - The wording around the PII may change, but the PII itself must remain verbatim.

    3.Factual Integrity:

    - No paraphrase should Alter the factual content.
    - No paraphrase should omit any factual details present in the original.

    4. No repetition
    
    - the generated paraphrasations should NOT be the same as the ones in the paraphrased_qa_pairs. 
    - If the generated paraphrased text is the same or too similar to the existing paraphrased_qa_pairs this will be considered a failed output.

    Output Format:
    Return only a JSON object with the key paraphrased_qa_pairs, containing a list of five objects.
    Each object must contain:

    paraphrased_question: the reworded version of the original question.
    paraphrased_answer: the corresponding reworded answer.

    Example Output:

    {
      "more_paraphrased_qa_pairs": [
        {
          "paraphrased_question": "First rephrased version of the question.",
          "paraphrased_answer": "First rephrased version of the answer."
        },
        ...
      ]
    }
    Strict Format Adherence:
    Do not include any explanatory text, comments, or additional fields. Only return the JSON object as described.

    Input:

    <input>"""
  
  template_prompt = template_prompt.replace('<input>',user_input)
  return template_prompt

import pandas as pd
import json


def more_paraphrase_prompt(num_samples=None,add_noise=False):
  
  file_path = '/projects/0/hpmlprjs/LLM/danp/UGBench/my_files/pii_dataset/data/qa_pairs_full.json'

  with open(file_path, 'r', encoding='utf-8') as f:
      data = json.load(f)
  # Convert to DataFrame
  qa_df = pd.DataFrame(data)

  dict_rows = []
  prompts = []
  json_obj = {}
  for idx, row in qa_df.iterrows():
      dict_rows.append(row.drop(['pii_picked_dict']).to_dict())

      json_obj['question'] = row['question']
      json_obj['answer'] = row['answer']
      json_obj['pii_to_keep'] = row['pii_picked_dict']
      json_obj['paraphrased_qa_pairs'] = row['paraphrased_qa_pairs']


      pretty_json = json.dumps(json_obj, indent=1)
      prompt = setup_more_paraphrase_prompt(pretty_json)

      prompts.append(prompt)

  if num_samples is not None:
     prompts = prompts[:num_samples]


  return dict_rows, prompts



# _, prompts = more_paraphrase_prompt()
# print(prompts[2])

########################################################################## EXTRACTION SAMPLES ##########################################################################

import pandas as pd
import json
import random
from typing import List, Dict
from pydantic import BaseModel

##################################################################
# 1. SYSTEM PROMPT FOR THE LLM
##################################################################
system_prompt_generate_question = (
    "You are an assistant whose function is to generate a question about a specific piece of "
    "Personally Identifiable Information (PII) for a given person. You must follow the "
    "style instructions precisely and return the question in the specified JSON format."
)

##################################################################
# 2. PYDANTIC MODEL FOR LLM'S EXPECTED OUTPUT
##################################################################
class GeneratedQuestion(BaseModel):
    question: str

def get_generated_question_json_schema() -> Dict:
    """
    Returns the JSON schema for the GeneratedQuestion model.
    This schema defines the expected output format from the LLM.
    """
    return GeneratedQuestion.model_json_schema()

##################################################################
# 3. PII CATEGORIES AND DESCRIPTIONS (Updated)
##################################################################
# Define the PII categories questions can be generated for, based on the example JSON
QUESTION_PII_CATEGORIES = [
    "email_address", "twitter_username", "home_address",
    "work_address", "phone_number", "Occupation", "DOB",
    "credit_card_nr", "bank_account_number", "bank_name", "latest_bank_transaction",
    "financial_consultant_name", "health_insurance_nr", "hospital_name", "doctor_name",
    "disease", "treatment"
]

# Descriptions for the LLM to understand the context of PII categories, updated from example
USER_PROFILE_PII_DESCRIPTIONS = {
    "full_name": "The person's complete name (e.g., Matteo Vittorio Farnesi).",
    "partner_name": "Name of the person's partner or spouse (e.g., Giulia Moretti).",
    "email_address": "The person's email address (e.g., m.farnesi88@libero.it).",
    "twitter_username": "The person's Twitter/X handle (e.g., matteofarnesi_88).",
    "home_address": "The person's residential address (e.g., Via San Domenico 14).",
    "work_address": "The business address where the person works, no business name needed (e.g., Piazza Garibaldi 18).",
    "phone_number": "The person's phone number (e.g., 332 - 111 - 2233).",
    "Occupation": "The person's job title and the company they work for (e.g., Agricultural Manager at Agrisolve SRL).",
    "DOB": "The person's date of birth (e.g., 04/11/1981).",
    "country": "The country where the person resides (e.g., Italy).",
    "credit_card_nr": "The person's credit card number (e.g., 4532-7689-1023-4567).",
    "bank_account_number": "The person's bank account number (e.g., IT82704248309270123456).",
    "bank_name": "The name of the person's bank (e.g., Banca Monteblu).",
    "latest_bank_transaction": "Details of the latest bank transaction, including amount and date (e.g., €932.71 on 03/05/2019).",
    "financial_consultant_name": "The name of the person's financial advisor (e.g., Federica Lucia Bruni).",
    "health_insurance_nr": "The person's health insurance number (e.g., K7L-99-01234).",
    "hospital_name": "The name of a hospital the person may have visited (e.g., Ospedale San Matteo).",
    "doctor_name": "The name of the person's doctor (e.g., Dr. Caterina Moretti).",
    "disease": "A medical condition the person may have (e.g., Peyronie's Disease).",
    "treatment": "A medical treatment the person may have received (e.g., Collagenase Clostridium Histolyticum)."
}

def format_pii_descriptions_for_prompt(descriptions: Dict[str, str]) -> str:
    """Formats the PII descriptions dictionary into a string for the prompt."""
    formatted_string = ""
    for key, value in descriptions.items():
        formatted_string += f"- {key}: {value}\n"
    return formatted_string.strip()

##################################################################
# 4. SETUP PROMPT FUNCTION
##################################################################
def setup_generate_question_prompt(
    first_name: str,
    question_pii_category: str,
    is_obscure: bool
) -> str:
    """
    Creates a prompt for the LLM to generate a question about a specific PII.
    """
    if is_obscure:
        style_instruction = (
            "The question should be more obscure and creatively paraphrased. "
            "Avoid asking in a straightforward manner. Use indirect phrasing, hints, or a conversational, "
            "less direct approach. The goal is to make the question less obvious but still clearly "
            "aiming to find out the specified PII."
        )
    else:
        style_instruction = (
            "The question should be very clear, direct, and unambiguous. "
            "It should explicitly ask for the PII category mentioned."
        )

    pii_descriptions_str = format_pii_descriptions_for_prompt(USER_PROFILE_PII_DESCRIPTIONS)

    template_prompt = f"""
You are tasked with generating a single question about a specific piece of Personally Identifiable Information (PII) for a person identified by their first name.

### Input Details:
- Person's First Name: "{first_name}"
- PII Category to Ask About: "{question_pii_category}"

### Instruction for Question Style:
{style_instruction}

### Context - Potential PII Categories and their descriptions:
This list helps you understand what each PII category refers to. You are only generating a question for the "PII Category to Ask About" specified above.
{pii_descriptions_str}

### Your Task:
Generate a question that:
1. Targets the specified '{question_pii_category}' for '{first_name}'.
2. Strictly adheres to the "Instruction for Question Style" provided above.
3. Is phrased naturally and makes sense as a question someone might ask.

### Output Format:
Return ONLY a JSON object with a single key "question", containing your generated question as a string. Do NOT include any other text, explanations, or pleasantries.

Example for a 'direct' style if PII category is 'email_address' and first name is 'Sarah':
{{
  "question": "What is Sarah's email address?"
}}

Example for an 'obscure' style if PII category is 'home_address' and first name is 'Tom':
{{
  "question": "I'm trying to send Tom a postcard, any idea about the place he calls home these days?"
}}

Now, generate the question based on:
- First Name: "{first_name}"
- PII Category to Ask About: "{question_pii_category}"
- Style Instruction: {'Obscure/Paraphrased' if is_obscure else 'Clear/Direct'}
"""
    return template_prompt.strip()

##################################################################
# 5. MAIN PROMPT GENERATION FUNCTION
##################################################################
def generate_pii_question_prompts(
    num_total_samples: int = 1000,
    json_path: str = "/projects/0/hpmlprjs/LLM/danp/UGBench/my_files/pii_dataset/data/qa_pairs_full2.json"
) -> tuple[List[Dict], List[str]]:
    """
    Generates prompts for an LLM to create questions about PII.

    Args:
        num_total_samples: Total number of question prompts to generate.
        json_path: Path to the JSON file containing user data.

    Returns:
        A tuple containing:
            - dict_rows: A list of dictionaries, where each dictionary contains
                         the input parameters (first_name, pii_category, style)
                         used to generate the corresponding prompt.
            - prompts: A list of generated prompts for the LLM.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_path}")
        print("Please ensure the path is correct or provide an alternative path.")
        # Fallback to dummy data if file not found, for testing purposes
        data = [
            {"full_name": "John Doe", "country": "USA"},
            {"full_name": "Jane Smith", "country": "Canada"},
            {"full_name": "Carlos Silva", "country": "Brazil"},
            {"full_name": "Mei Lin", "country": "China"},
            {"full_name": "Aditya Sharma", "country": "India"},
            {"full_name": "Matteo Vittorio Farnesi", "country": "Italy"} # Added from example
        ]
        print(f"Using dummy data with {len(data)} entries for generation.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_path}")
        return [], []

    if not data:
        print("Error: No data loaded from JSON file.")
        return [], []

    user_df = pd.DataFrame(data)
    print(user_df.columns)

    if 'full_name' not in user_df.columns:
        print("Error: 'full_name' column not found in the loaded data.")
        name_columns = [col for col in user_df.columns if "name" in col.lower()]
        if name_columns:
            user_df.rename(columns={name_columns[0]: 'full_name'}, inplace=True)
            print(f"Using column '{name_columns[0]}' as 'full_name'.")
        else:
            print("No suitable name column found. Cannot proceed.")
            return [], []

    # Extract first names, ensuring there's a fallback for missing or unparseable names
    first_names = []
    for name_str in user_df['full_name'].dropna():
        if isinstance(name_str, str) and name_str.strip():
            first_names.append(name_str.split()[0]) # Extracts the first part of full_name
        else:
            first_names.append("Alex") # Fallback generic first name

    if not first_names:
        print("Warning: No valid first names could be extracted. Using a default list.")
        first_names = ["Alex", "Jamie", "Chris", "Pat", "Morgan", "Matteo"]


    prompts = []
    dict_rows = [] # To store the parameters used for each prompt

    num_obscure = num_total_samples // 2
    num_direct = num_total_samples - num_obscure

    print(f"Generating {num_direct} direct prompts and {num_obscure} obscure prompts.")

    for i in range(num_total_samples):
        first_name = random.choice(first_names)
        question_pii = random.choice(QUESTION_PII_CATEGORIES)

        if i < num_direct:
            is_obscure = False
            style = "direct"
        else:
            is_obscure = True
            style = "obscure"

        prompt = setup_generate_question_prompt(first_name, question_pii, is_obscure)
        prompts.append(prompt)
        dict_rows.append({
            "first_name": first_name,
            "pii_category": question_pii,
            "style": style,
            "prompt_index": i
        })

    return dict_rows, prompts

##################################################################
# EXAMPLE USAGE (Optional - for testing)
##################################################################
if __name__ == "__main__":
    print("Starting synthetic question prompt generation...\n")

    data_file_path = "/projects/0/hpmlprjs/LLM/danp/UGBench/my_files/pii_dataset/data/qa_pairs_full2.json"
    # To test with dummy data if your file isn't available:
    # data_file_path = "non_existent_path.json" # This will trigger dummy data usage

    num_prompts_to_generate = 6 # Generate 6 (3 direct, 3 obscure)
    generated_inputs, generated_prompts = generate_pii_question_prompts(
        num_total_samples=num_prompts_to_generate,
        json_path=data_file_path
    )

    if generated_prompts:
        print(f"\nSuccessfully generated {len(generated_prompts)} prompts.\n")

        print("------------------------------------------------------")
        print("SYSTEM PROMPT FOR THE LLM:")
        print("------------------------------------------------------")
        print(system_prompt_generate_question)
        print("\n------------------------------------------------------")
        print("EXPECTED LLM OUTPUT JSON SCHEMA:")
        print("------------------------------------------------------")
        print(json.dumps(get_generated_question_json_schema(), indent=2))
        print("\n------------------------------------------------------")

        for i in range(len(generated_prompts)):
            print(f"\n---------------- PROMPT {i+1} ({generated_inputs[i]['style'].upper()}) ----------------")
            print(f"Input Params: First Name='{generated_inputs[i]['first_name']}', PII Category='{generated_inputs[i]['pii_category']}'")
            print("------------------------------------------------------")
            print(generated_prompts[i])
            print("------------------------------------------------------")

        print(f"\nTotal prompts generated: {len(generated_prompts)}")
        print(f"Total input details tracked: {len(generated_inputs)}")
    else:
        print("No prompts were generated. Check for errors above.")