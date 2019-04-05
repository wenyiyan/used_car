#!/usr/bin/python
# -*- encoding: utf-8 -*-


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
 		
 		print user_input

		yyyy_mm = user_input['age_month'][0]       #2016-01
		print yyyy_mm 
		yyyy_mm_dd = yyyy_mm +'-01'                 # 2016-01-01
		print yyyy_mm_dd
		bought_date = pd.to_datetime(yyyy_mm_dd)    # 2016-01-01
		print bought_date
		age_month = round((datetime.now() - bought_date)/np.timedelta64(1, 'M'),0)
		age_month = str(int(age_month))
		user_input['age_month'] = [age_month]     # 20 -> '20' -> ['20']
		print user_input['age_month']


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

		# get display text
		user_input_age_month = int(user_input['age_month'][0])     # convert one list of string to string
		user_input_color = user_input['color'][0]
		user_input_miles = int(user_input['miles'][0])           
		user_input_province = user_input['province'][0]         
		user_input_source = user_input['source'][0]            
		user_input_transmission = user_input['transmission'][0]            
		user_input_year = int(user_input['year'][0])    

		# get display text lookup table
		transmission_lookup = {'AT' : u'自排','MT' : u'手排'}
		color_lookup = {'white' : u'白色','black' : u'黑色','silver' : u'银色','golden' : u'金色','brown' : u'棕色'}
		source_lookup = {'merchandise_source' : u'委托线下商户','individual_sale' : u'个人售卖','platform_source' : u'线上平台'}
		province_lookup = {'anhui': u'安徽','beijing': u'北京','chongqing': u'重庆','fujian': u'福建','gansu': u'甘肃','guangdong': u'广东','guangxi': u'广西','guizhou': u'贵州','hainan': u'海南','hebei': u'河北','heilongjia': u'黑龙江','henan': u'河南','hubei': u'湖北','hunan': u'湖南','jiangsu': u'江苏','jiangxi': u'江西','jilin': u'吉林','liaoning': u'辽宁','neimenggu': u'内蒙古','ningxia': u'宁夏','qinghai': u'青海','shandong': u'山东','shanghai': u'上海','shannxi': u'陕西','shanxi': u'山西','sichuan': u'四川','tianjin': u'天津','tibet': u'西藏','xinjiang': u'新疆','yunnan': u'云南','zhejiang': u'浙江'}       

		# get display text lookup table to get final dispaly words
		user_input_age_month_display = user_input_age_month 
		user_input_color_display =  color_lookup[user_input_color]
		user_input_miles_display = user_input_miles
		user_input_province_display = province_lookup[user_input_province]
		user_input_source_display = source_lookup[user_input_source]
		user_input_transmission_display = transmission_lookup[user_input_transmission]
		user_input_year_display = user_input_year


		# prediction array
		data = [origin_data]
		my_prediction = clf.predict(data)
		my_prediction = round(my_prediction,2)
	return render_template('green_result.html',prediction = my_prediction,age_month = user_input_age_month_display,color = user_input_color_display,miles = user_input_miles_display,province = user_input_province_display,source = user_input_source_display, transmission = user_input_transmission_display, year = user_input_year_display)


if __name__ == '__main__':
	app.run(debug=True)