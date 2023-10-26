import requests 
url = 'http://peng24.pythonanywhere.com/predict' 

client = {"education":"bachelors",
"joiningyear":2018,
"city":"pune",
"paymenttier":"third_tier",
"age":32,
"gender": "male",
"everbenched":"yes",
"experienceinCurrentdomain":5
}
result=requests.post(url, json=client).json()

print(result)
   