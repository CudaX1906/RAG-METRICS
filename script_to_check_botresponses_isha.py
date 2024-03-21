import json
import os
import sys
import tempfile
import time
import traceback
import uuid

import openai
import pandas as pd
from tqdm import tqdm
import requests

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
TEMP_FOLDER = tempfile.mkdtemp()
print(os.getcwd())


# To get the URL based on the environment
def url_based_on_env(bot_record_id, env):
    environment = str(env).lower()

    if environment == "qa":
        return f"https://alpha-qa-bot-builder-agents.skil.ai/v1/nlu/{bot_record_id}/message-agent"
    elif environment == "dev":
        return f"https://alpha-bot-builder-agents.skil.ai/v1/nlu/{bot_record_id}/message-agent"
    elif environment == "prod":
        return f" https://gamma-bot-builder-agents.skil.ai/v1/nlu/{bot_record_id}/message-agent"
    else:
        # Handle unknown environment here, for example, raise an exception or return a default URL
        raise ValueError(f"Unknown environment: {environment}")


# To trigger a mail after the report is generated.
def trigger_botbuilder_report(output_excel_path, details, toMail):
    url = "https://dev-01.skil.ai/ai-email-template/v0/send/email-template-with-attachments"

    payload = {'subject': 'This mail is regarding Isha bot responses',
               'emailTemplateId': '65e258d0551a940023d387f6',
               'fromEmail': 'quality-analysis@skil.ai',
               'toEmail': 'quality-analysis@skil.ai,prod_support@skil.ai',
               'details': details}
    files = [
        ('file', ('botResponsesReport.xlsx', open(output_excel_path, 'rb'),
                  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'))
    ]
    headers = {}

    response = requests.request("POST", url, headers=headers, data=payload, files=files)

    print("in func trig", response.text)


# trigger error mail
def trigger_error_mail(error, tomail, env):
    url = "https://dev-01.skil.ai/ai-email-template/v0/send/email-template"

    payload = json.dumps({
        "emailTemplateId": "65ade3c0c7340f002fb83781",
        "fromEmail": "quality-analysis@skil.ai",
        "toEmail": [
            tomail
        ],
        "subject": f'QA Services:Bot Responses Report in {env} environment-Docubaat',
        "error": error
    })
    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)


# To get the url based on the environment to get transaction_details
def url_based_on_environment_for_transactionId(env, transactionId):
    if env == "qa":
        url = f"https://alpha-bot-builder-agents.skil.ai/v1/deep-learning/generative/web-qa/info-by-transaction-id?id={transactionId}"
        print("url to get details based on transactionId-->", url)
        return url
    else:
        url = f"https://gamma-bot-builder-agents.skil.ai/v1/deep-learning/generative/web-qa/info-by-transaction-id?id={transactionId}"
        print("url to get details based on transactionId-->", url)
        return url


# Funtion to get transaction details
def details_from_transactionId(transactionId, env):
    env = str(env).lower()
    try:
        url = url_based_on_environment_for_transactionId(env, transactionId)
        print(url)
        payload = ""
        headers = {
            'Content-Type': 'application/json'
        }

        response = requests.request("GET", url, headers=headers, data=payload)
        if response.status_code == 200:
            # Parse the JSON response
            response_data = response.json()

            # Extract the prompt value if available
            prompt_value = response_data.get('data', {}).get('latest_message', {}).get('webQaResponse', {}).get(
                'prompt',
                None)
            context = response_data.get('data', {}).get('latest_message', {}).get('webQaResponse', {}).get(
                'currentConversation',
                None)
            context = "\n".join([c["content"] for c in context if c["role"] == "function"])

            ps = prompt_value.split("---")
            prompt_value = ps[0] + "\n---\nContext:\n" + context + "\n---\n" + ps[2]

            if prompt_value is not None:
                # Print the extracted value
                print("Prompt Value:", prompt_value)
                return prompt_value
            else:
                print("Prompt value not found in the response.")
                return None
        else:
            print(f"Error: {response.status_code}")
        return None
    except:
        prompt_value = "Null"
    return context


"""Setting the Environment Variables

"""

"""
As we are depending on openAI for the Validation of Bot Responses. We need their API Key inorder to use their Services.Please Maintain This Key as secret one.
"""
openai.api_key = os.environ.get("OPENAI_API_KEY") or "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"

"""
This function Hits the message Agent of the respective Environment and gets the message.
"""


# Get response from based on the query(msg) from the backend API for responses
def url_response(url, msg, user_id):
    payload = json.dumps({
        "message": msg,
        "user_id": user_id
    })
    headers = {
        'Content-Type': 'application/json'
    }
    try:
        response = requests.request("POST", url, headers=headers, data=payload)
        print("In func url", response)
        get_response = response.json()
        responses = get_response['bot_responses']#[0]['text']
        print(responses)
        transaction_id = get_response['transactionId']
        result = [responses, transaction_id]
        return result
    except Exception as e:
        print(e)
        return None, None
    return None


_generative_qa_excel_func = {
    "name": "handle_test_case",
    "description": "Depending on the status, similarity score (and the reasoning) of the test-case, "
                   "take actionable next steps",
    "parameters": {
        "type": "object",
        "properties": {
            "status": {
                "type": "boolean",
                "description": "boolean flag indicating whether the test-case has passed or failed"
            },
            "score": {
                "type": "number",
                "description": "a score between 0 - 1, indicating the similarity score between the "
                               "expected response and the bot response. 1 is a perfect pass and 0 is a perfect fail"
            },
            "explanation": {
                "type": "string",
                "description": "a detailed explanation on why the test-case has passed or failed, as well as its score"
            }
        }
    }
}

"""
OPENAI PROMPT FOR PERFORMING ANALYSIS TASK
"""

_generative_qa_excel_prompt = (

    f"You are an extremely skilled QA Analyst that specializes in validating a chat bot's outputs. "
    f"You will be provided with:"
    f"   1. Query: This is the question the user asked the chat bot"
    f"   2. Expected Response: This is the actual response that the bot is expected to say to the user."
    f"                         This message has been curated by an expert in the chat bot's domain."
    f"   3. Bot Response: This is what the bot responded to the user. While it is not expected for the bot's "
    f"                    response to completely match the expected response, it should ideally have a similar overall structure and convey the same message."
    f"\n\nYour task is to carefully consider both the expected and bot responses and validate the bot response."
    f"Think step-by-step and validate if the test-case has passed or failed."
    f"Also provide an accurate score which quantifies the bot response's overall similarity to the expert-curated expected response."
    f"Also provide an explanation on your thought process and validation steps. Provide clear reasoning on why the test-case has passed or failed, and why you assigned a certain score to the test-case.")

"""

"""

"""


"""
"""
compare the expected response and bot response.
"""


def process_all_rows(query, expected_response, bot_response):
    message = [{"role": "user",
                "content": f"User's question:\n\n{query}"
                           f"\n\n###\n\n###Expected Response:\n\n{expected_response}"
                           f"\n\n###\n\nBot Response:\n\n{bot_response}"}]

    try:
        response = openai.ChatCompletion.create(
            messages=[{"role": "system", "content": _generative_qa_excel_prompt}] + message,
            temperature=0,
            functions=[_generative_qa_excel_func],
            function_call={"name": _generative_qa_excel_func["name"]},
            model="gpt-3.5-turbo-16k",
        )

        llm_prediction = json.loads(response.choices[0]["message"]["function_call"]["arguments"])
        return llm_prediction.values()

    except Exception as e:
        traceback.print_exc()

    return None, None, None


"""# Action Required
Setting The Bot Details on which You Would Like To Perform Test.


*   **BotRecordId** : The BotRecordId of the Bot. For Example (**655dd6621e322460fb64e2e1**)
*   **Message Agent URL** : It will consists of two fields.

            Environment - The Environment You Would Perform Testing
            BotRecordId - The BotRecordId of the Bot You Would Like to perform




"""

# botId: s_f4d17b -- version : 0.YCADL -- Bot Rec Id : 655dd6621e322460fb64e2e1

# CHOOSE THE BOT RECORD ID FOR THE BOT HERE BOT RECORD ID REMAINS SAME FOR BOTH DEV AND QA ENVIRONMENT
# REPLACE REQUIRED BOT REC ID

"""CHOOSE THE ENVIRONMENT ---

-  DEV------------------------- FOR ML-DEV-SERVER
-  QA --------------------------FOR QA-ENVIRONMENT

"""

"""
This Function reads Input Excel and Performs all actions and generate a report.
"""


def process_qa_excel_for_multiple_bots(excel_path, bot_record_id, environment, tomail):
    try:
        # bot_name = get_bot_name(bot_record_id)
        # Get the Query from Excel and send to bot
        """
    Make Sure the input Excel has the Following Columns:
    1. Query             - The Question
    2. Expected Response - Your Expected Response
    3. Conversation Id  -- Make Sure if the Questions to be Asked as Follow Up Should have Same Flow ID
    """
        formatted_data = []
        excel_data = pd.read_excel(excel_path)
        url = url_based_on_env(bot_record_id, environment)
        print(url)
        FLOW_CACHE = {}
        for _, row in tqdm(excel_data.iterrows(), total=len(excel_data)):
            query = row["Query"]
            expected_response = row["Expected Response"]
            flow_id = row["Conversation Id"]
            if not pd.isna(flow_id):
                user_id = FLOW_CACHE.get(flow_id, str(uuid.uuid4()))
                FLOW_CACHE[flow_id] = user_id
            else:
                user_id = str(uuid.uuid4())
            result = url_response(url, query, user_id)
            bot_response = result[0]
            transaction_id = result[1]
            print(transaction_id)
            print(bot_response)
            prompt = details_from_transactionId(transaction_id, environment)
            # with open("res.json", "a") as res:
            #    json.dump(json.dumps(prompt, indent=2), res, indent=2)
            status, score, explanation = process_all_rows(query, expected_response, bot_response)

            current_data = {**row, "bot response": bot_response, "transaction id": transaction_id, "prompt": prompt,
                            "Status": status, "score": score,
                            "Explanation for Test result": explanation}
            formatted_data.append(current_data)
        return formatted_data
    except Exception as e:
        # Save the error message in a variable
        error_message = str(e)
        print(f"An error occurred: {error_message}")
        trigger_error_mail(error_message, tomail, environment)
        # Optionally, you can re-raise the exception if needed
        raise e


if __name__ == "__main__":
    # Example usage without hardcoded values
    input_file_path = input("Enter input file path: ")  # Replace this with actual user input or form data
    bot_record_id = input("Enter bot record ID: ")
    environment = input("Enter environment: ")
    tomail = input("Enter recipient email: ")
    doc_details = input("Enter the details of the doc")
    bot_name = input("Enter botName")
    process_qa_excel_for_multiple_bots(input_file_path, bot_record_id, environment, tomail)
