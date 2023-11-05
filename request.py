import requests

url="http://localhost:9696/predict"

sample = {"work_year": "2020", "experience_level": "en", "employment_type": "ct",
          "job_title":"data_scientist	", "employee_residence":"us", "remote_ratio":"0",
            "company_location":"us", "company_size":"m"}

response = requests.post(url, json=sample).json()
print(response)
