import joblib
import streamlit as st
from script_to_check_botresponses_isha import url_based_on_environment_for_transactionId,url_based_on_env,url_response,details_from_transactionId
import re
import uuid
import requests

# Initialize load_model attribute
st.session_state.load_model = joblib.load('logistic_regression_model.pkl')

def extract_reference_text(input_text):
    pattern = r'Reference\s+ID\s+(\d+):\s*(.*?)(?=(?:Reference\s+ID\s+\d+|$|\-{3}))'
    
    regex = re.compile(pattern, re.DOTALL)
    
    matches = regex.findall(input_text)
    reference_texts = [match[1].strip() for match in matches]
    
    return reference_texts

def request_metric_api(api_url, data):
    try:
        response = requests.post(api_url, json=data)
        response.raise_for_status()
        result = response.json()
        return result
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Error calling API: {str(e)}")

question = st.text_input("Enter the Question:")


if question:
    bot_record_id = "652d06306ea64c8e2a961c27"
    url = url_based_on_env(bot_record_id, "qa")
    user_id = str(uuid.uuid4())
    result = url_response(url, question, user_id)
    bot_response = result[0]
    transactionId = result[1]
    
    prompt = details_from_transactionId(transactionId, "qa")
    if prompt:
        contexts = extract_reference_text(prompt)
        input_data = {
                "questions": [question],
                "contexts": [contexts],
                "answers": [bot_response[0]["text"]]
            }
        
        answer_relevancy_Score = request_metric_api("http://127.0.0.1:8000/answerrelevance/", input_data)
        context_utilization_score = request_metric_api("http://127.0.0.1:8000/contextprecision/", input_data)
        context_relevancy =  request_metric_api("http://127.0.0.1:8000/contextrelevancy/", input_data)
        faithfulness_score = request_metric_api("http://127.0.0.1:8000/faithfulness/", input_data)

        st.write(f"Answer Relevancy Score:{answer_relevancy_Score}")
        st.write(f"Context Utilization Score :{context_utilization_score}")
        st.write(f"Context Relevancy Score :{context_relevancy}")
        st.write(f"Faithfulness Score :{faithfulness_score}")
        st.write(input_data["answers"])
        response = st.session_state.load_model.predict([[answer_relevancy_Score,context_utilization_score,context_relevancy,faithfulness_score]])
        if response==1:
            st.write("Answer is True")
        else:
            st.write("Answer is False")
        
    else:
        st.write(prompt)
