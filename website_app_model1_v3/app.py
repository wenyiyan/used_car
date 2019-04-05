from flask import Flask,render_template,url_for,request
from flask_bootstrap import Bootstrap 
import pandas as pd 
import numpy as np 
from werkzeug.datastructures import ImmutableMultiDict

# ML Packages
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib


app = Flask(__name__)
Bootstrap(app)


@app.route('/')
def index():
	return render_template('green.html')

@app.route('/regression', methods=['POST'])
def regression():
	#df= pd.read_csv("data/names_dataset.csv")
	## Features and Labels
	#df_X = df.name
	#df_Y = df.sex
    
    # Vectorization
	#corpus = df_X
	#cv = CountVectorizer()
	#X = cv.fit_transform(corpus) 
	aa = [0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,17,0,0,0,1,1,1,0]

	# Loading our ML Model
	naivebayes_model = open("models/pricepredictmodel.pkl","rb")
	clf = joblib.load(naivebayes_model)

	# Receives the input query from form
	####################################################    please 	array   ############################################
	if request.method == 'POST':
		name = request.form
		car_year = request.form['year']
		car_transmission = request.form['transmission']
		namequery = aa
		data = [namequery]
		#vect = cv.transform(data).toarray()
		my_prediction = clf.predict(data)
	return render_template('results.html',prediction = my_prediction,name = name,transmission = car_transmission,year = car_year)


if __name__ == '__main__':
	app.run(debug=True)