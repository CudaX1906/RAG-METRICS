import csv
import json
import requests

# Read CSV file
csv_file = "dataset.csv"

# with open(csv_file, "r") as file:
#     reader = csv.DictReader(file)
#     dataset = {"items": {

#     }}

# Convert to JSON
data_from_csv = {
    "items": [
        {
            "name": "item1",
            "description": "Description for item 1",
            "price": 10.5,
            "quantity": 2
        },
        # Add more items as needed
    ]
}

# Convert to JSON
json_payload = json.dumps(data_from_csv)

# Send POST request to FastAPI endpoint
url = "http://127.0.0.1:8000/process_dataset"
headers = {"Content-Type": "application/json"}

response = requests.post(url, headers=headers, data=json_payload)

# Print the response
# print(response.status_code)
print(response.json())
