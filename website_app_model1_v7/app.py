from flask import Flask,render_template,url_for,request
from flask_bootstrap import Bootstrap 
import pandas as pd 
import numpy as np 
from werkzeug.datastructures import ImmutableMultiDict
from datetime import datetime
import sys

# ML Packages
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib


app = Flask(__name__)
Bootstrap(app)


@app.route('/')
def index():
	return render_template('index.html')


@app.route('/green')
def green():
	return render_template('green.html')


@app.route('/regression', methods=['POST'])
def regression():

	# Loading our ML Model
	naivebayes_model = open("models/pricepredictmodel.pkl","rb")
	clf = joblib.load(naivebayes_model)
	origin_data =  [0] * 53
	index_lookup = {'2007':0,'2008':1,'2009':2,'2010':3,'2011':4,'2012':5,'2013':6,'2014':7,'2015':8,'2016':9,'2017':10,'2018':11,'2019':12,'miles':13,'MT':14,'beijing':15,'chongqing':16,'fujian':17,'gansu':18,'guangdong':19,'guangxi':20,'guizhou':21,'hainan':22,'hebei':23,'heilongjia':24,'henan':25,'hubei':26,'hunan':27,'jiangsu':28,'jiangxi':29,'jilin':30,'liaoning':31,'neimenggu':32,'ningxia':33,'qinghai':34,'shandong':35,'shanghai':36,'shannxi':37,'shanxi':38,'sichuan':39,'tianjin':40,'tibet':41,'xinjiang':42,'yunnan':43,'zhejiang':44,'age_month':45,'brown':46,'golden':47,'silver':48,'white':49,'publish_date':50,'merchandise_source':51,'platform_source':52}
	
	# Receives the input query from form
	if request.method == 'POST':
		user_input = request.form.to_dict(flat=False)
		
		## dropdown_box:
		for i in user_input:  #key: province, color  is list
			feature = user_input[i][0]  #first element  [u'beijing']
			if feature in index_lookup:  # if beijing in index lookup dectionary  #print feature
				index = index_lookup[feature]
				origin_data[index] = 1


		## input box:
		origin_data[45] = int(user_input['age_month'][0])  #age_month
		origin_data[13] = int(user_input['miles'][0])/10000      #mile
		origin_data[50] = int(datetime.now().strftime('%m'))


		#car_year = request.form['year']
		#car_transmission = request.form['transmission']
		data = [origin_data]
		my_prediction = clf.predict(data)
		my_prediction = round(my_prediction,2)
	return render_template('green_result.html',prediction = my_prediction,name = user_input,transmission = 'transmission',year = 'year')


if __name__ == '__main__':
	app.run(debug=True)