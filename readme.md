# Project Scope
This project is for a classification model to detect whether the employee will leave the company or not. \
we used data Employee.csv

#### Data Preparation
We prepared the data by: \
  1- having lowercase strings for column names. \
  2- For string columns we also converted it to camel case. \
After That, we divided the data to train, validate, and test data. \
Then we tried different models and chose the best model.
we used the ROC_AUC to compare the models.

Using pickle we were able to export the chosen model to be used.

## Deploying the model locally 

We created a Dockerfile to create an image. \
firstly we can build the model using: \
docker build . -t emp_model
then we can run the image using: \ 
docker run   -it  -p 9696:9696 emp_model 

then we can run the predict_local.py after installing requests in our environment \
as the URL will be: \
url = 'http://localhost:9696/predict' \ 
and run \
python predict_local.py \

# Using the deployed model 
to use the deployed model the URL will be: \
url = 'http://peng24.pythonanywhere.com/predict' \
then run \
python predict.py \


# predicted dictionary  shape
Make sure the predicted dictionary to be in this form: \
client = {"education":"bachelors", \
"joiningyear":2018, \
"city":"pune", \
"paymenttier":"third_tier", \
"age":32, \
"gender": "male", \
"everbenched":"yes", \
"experienceinCurrentdomain":5 \
} \
all strings should be in lowercase.








