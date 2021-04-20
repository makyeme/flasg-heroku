import flasgger
from flasgger import Swagger
from flask import Flask, request
import pickle
import pandas as pd
import numpy as np

#Initiate app
app = Flask(__name__)
Swagger(app)



#Load regression model
pickle_in = open('linreg.pkl', 'rb')
linreg = pickle.load(pickle_in)


#Launching welcome page
@app.route('/')
def Welcome():
   return "welcome all"


#creating prediction function for salary
@app.route('/predict')
def predictSalary():

#craeting a swagger API
    '''API to predict salary based on number of experience years
    ---
    ---
    parameters:
        - name: years
          in: query
          type: number
          required: true
    responses:
          200:
              description: The output values

    '''
    years = request.args.get('years')
    prediction = linreg.predict([[years]])
    return 'Your predicted salary is:'+ str(prediction)




#creating prediction for the values in csv file
@app.route('/predict_file',methods = ["POST"])
def predictFile():

    '''API to predict salary based on number of experience years
    ---
    parameters:
        - name: file
          in: formData
          type: file
          required: true
    responses:
          200:
              description: The output values
    '''


    df = pd.read_csv(request.files.get('file'))
    prediction = linreg.predict(df)
    return 'Your predicted salary values for csv are:'+ str(list(prediction))



    
    
if __name__ == '__main__':
   app.run(debug=True, port=5001)