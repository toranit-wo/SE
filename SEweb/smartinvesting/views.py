from django.shortcuts import render
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import pandas_datareader as pdr
import datetime
import pandas as pd
import numpy as np

# Create your views here.
def index(req):

    return render(req,'indexT.html',{})

def cal(req):
    out=dict()
    if req.method == 'POST':
        data = req.POST
        ss = data['Sday']
        se = data['Lday']
        s =  datetime.datetime.strptime(ss, '%Y-%m-%d') 
        e =  datetime.datetime.strptime(se, '%Y-%m-%d' )  
        h = data['set50'] + '.BK'
        poly_model = make_pipeline(PolynomialFeatures(5), LinearRegression())
        pttgc = pdr.get_data_yahoo(h.upper(), start = datetime.datetime(1999, 9, 5 ), end = datetime.date.today())
        kk = pttgc[['Adj Close']]
        kk = kk.values
        yt = kk.reshape(len(kk))
        xt = pttgc.index
        xt = xt.map(datetime.datetime.toordinal)
        my = xt.values
        xt = np.reshape(my, (len(kk), 1))
        poly_model.fit(xt,yt) 
        table = []
        s = s.toordinal()
        e = e.toordinal()
        # xf = [[i] for i in range(s,e+1)]
        table = [[datetime.date.fromordinal(i).strftime('%A %d-%m-%Y'),'-' if len(pttgc[pttgc.index.date==datetime.date.fromordinal(i)]['Adj Close'].values)==0 else round(pttgc[pttgc.index.date==datetime.date.fromordinal(i)]['Adj Close'].values[0],3) ,round(poly_model.predict([[i]])[0],3)]  for i in range(s,e+1)]
        print(table)
        out['result'] = table
        out['set'] = data['set50'].upper()
        print()
        return render(req,'cal.html',out)
    else : 
        return render(req,'cal.html',out)
# def compare(req):
#     return render(req,'compare.html',{})

