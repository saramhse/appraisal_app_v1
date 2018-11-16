from flask import render_template, request, flash, redirect, Flask
from forms import HouseForms, train_flask, min_built
import pickle
import dill
import pandas as pd
import os.path
import os
import numpy as np
import pandas as pd
# from flask_wtf.csrf import CSRFProtect
# from config import SECRET_KEY

app = Flask(__name__)

irvine_ridge_model = open("irvine_ridge_model.pkl","rb")
irvine_ridge = pickle.load(irvine_ridge_model)

tustin_ridge_model=open("tustin_ridge_model.pkl","rb")
tustin_ridge=pickle.load(tustin_ridge_model)

costamesa_model=open("costamesa_model.pkl","rb")
costamesa_ridge=pickle.load(costamesa_model)

# lakeforest_model=open("lakeforest_model.pkl","rb")
# lakeforest_ridge=pickle.load(lakeforest_model)

test_model = open("ridge_mode.pkl","rb")
test_ridge_model = pickle.load(test_model)

def default_none(input_data):
    if input_data != None:
        return input_data
    else:
        return None

@app.route("/")
def homepage():
    return render_template("index.html")

@app.route("/map")
def mapper():
    return render_template("orangemap.html")

@app.route("/priceForm")
def root():

    # if city_entered=="Irvine" or city_entered=="irvine":
    #     global irvine_ridge
    # else city_entered=="Tustin" or city_entered="tustin":
    #     global tustin_ridge

    form = HouseForms(csrf_enabled=False)  
    return render_template("index1.html",
    title = "House Price Prediction",
    form = form)

#send info from fron end to back end : post.... get: requesting data
@app.route("/calculate", methods = ["POST"])
def index():

    # global irvine_ridge
    form = HouseForms(csrf_enabled=False)
    # city=[str(form.city.data)]
    city=str(form.city.data)
    year_entered=int(form.yearBuilt.data)

    print(city)
    print('hello')

    n=train_flask()

    minimums=min_built()

    city_min=0
    bin_sample=0

    if city=="Irvine" or city=="irvine":
        city_min=minimums[0]
        bin_sample=n[0]['train_built'].unique()
        # ['train_built']
    if city=="tustin" or city=='Tustin':
        city_min=minimums[1]
        bin_sample=n[2]['train_built'].unique()
    if city=="Costa Mesa" or "Costa mesa" or "costa mesa":
        city_min=minimums[2]
        bin_sample=n[4]['train_built'].unique()
    # if city=="Lake Forest" or "Lake forest" or "lake forest":
    #     city_min=minimums[3]
    #     bin_sample=n[6]['train_built'].unique()

    print(minimums)
    print(city_min)
    print(bin_sample)

    binned_yr=round((year_entered-city_min)/10,0)

    #closest existing bin to binned_yr

    def my_min(sequence):
        low = sequence[0] # need to start with some value
        for i in sequence:
            if i < low:
                low = i
        return low

    diff=[]
    for i in bin_sample:
        x=abs(i-binned_yr)
        diff.append(x)
        
    min_difference=my_min(diff)
    bin_index=diff.index(min_difference)
    right_bin=bin_sample[bin_index]
        
    print(type(right_bin))

    user_dictionary={
        'zip':[str(int(form.zipcode.data))],
        'train_built':[str(right_bin)],
        'type':[str(form.buildingType.data)],
        'beds':[float(form.bedrooms.data)],
        'baths':[float(form.bathrooms.data)],
        'sqrft':[float(form.Squarefeet.data)],
        'lot':[float(form.lotsize.data)]}
        # 'city':[city]}
        # [str(form.city.data)]}
    print(f'user dictionariy: {user_dictionary}')
    # 'train_built':[str(form.yearBuilt.data)]

    user_df=pd.DataFrame(user_dictionary)
    user_df_fit=pd.get_dummies(user_df,columns=['zip','type','train_built'])

    # n=train_flask()

    # prediction=0
    if city=="irvine" or city=="Irvine":
        for i in n[1].columns:
            if i in user_df_fit.columns:
                pass
            else:
                user_df_fit[i]=0
        prediction = int(np.expm1(irvine_ridge.predict(user_df_fit)))

    
    if city=="tustin" or city=="Tustin":
        for i in n[3].columns:
            if i in user_df_fit.columns:
                pass
            else:
                user_df_fit[i]=0
        prediction = int(np.expm1(tustin_ridge.predict(user_df_fit)))
    
    if city=="Costa mesa" or city=="Costa Mesa" or city=="costa mesa":
        for i in n[5].columns:
            if i in user_df_fit.columns:
                pass
            else:
                user_df_fit[i]=0
        prediction = int(np.expm1(costamesa_ridge.predict(user_df_fit)))
    
    # if city=="Lake Forest" or city=="Lake forest" or city=="lake forest":
    #     for i in n[7].columns:
    #         if i in user_df_fit.columns:
    #             pass
    #         else:
    #             user_df_fit[i]=0
    #     prediction = int(np.expm1(lakeforest_ridge.predict(user_df_fit)))


    print(user_df_fit)
    print(user_df_fit.columns)
    print(n[1].columns)
    # print(prediction)


    # prediction = int(np.expm1(tustin_ridge.predict(user_df_fit)))

    # prediction = int(np.expm1(tustin_ridge.predict(user_df_fit)))

    # print([str(int(form.zipcode.data))])
    # print([str(binned_yr)])
    # print([str(form.buildingType.data)])
    # print([float(form.bedrooms.data)])
    # print([float(form.bathrooms.data)])
    # print([float(form.Squarefeet.data)])
    # print([float(form.lotsize.data)])
    # print(n)

    # print(prediction)

    
    return render_template("index1.html", title = "House Price Prediction", form = form, prediction = '${:,.2f}'.format(prediction))

if __name__ == ("__main__"):
    app.run(debug=True,port=7700)