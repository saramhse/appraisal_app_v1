from flask_wtf import FlaskForm
from wtforms import FloatField, SelectField, StringField
from wtforms.validators import DataRequired,optional,NumberRange
import pickle
import numpy as np
import pandas as pd

building_type = [("sfr","Single Family Residence"),('condo', 'Condominium'),('thr', 'Townhome Residence'),('mfr', 'Multifamily Residence') ]
bedrooms_choice = [(1,1),(2,2),(3,3),(4,4),(5,5)]
bathrooms_choice = [(1,1),(2,2),(3,3),(4,4)]
zipcode_choice=[("92626","Costa Mesa 92626"),("92627","Costa Mesa 92627 "),("92660","Costa Mesa 92660"),
                ("92663","Costa Mesa 92663"),("92704","Costa Mesa 92704"),("92707","Costa Mesa 92707"),
                ("90620","Irvine 90620"),("91618","Irvine 91618"),("92602","Irvine 92602"),
                ("92603","Irvine 92603"),("92604","Irvine 92604"),("92606","Irvine 92606"),("92612","Irvine 92612"),
                ("92614","Irvine 92614"),("92618","Irvine 92618"),("92620","Irvine 92620"),("92630","Irvine 92630"),
                ("92653","Irvine 92653"),("92657","Irvine 92657"),("92782","Irvine 92782"),
                ("92606","Tustin 92606"),("92602","Tustin 92602"),("92705","Tustin 92705"),
                ("92780","Tustin 92780"),("92782","Tustin 92782"),("92867","Tustin 92867")]

# costamesa='92626', '92627', '92660','92663', '92704', '92707'
# irvine= '90620', '91618', '92602',
    #    '92603', '92604', '92606', '92612', '92614',
    #    '92618', '92620', '92630', '92653', '92657',
    #    '92782'
# tustin='92606', '92602', '92705', '92780', '92782',
#        '92867', 

class HouseForms(FlaskForm):
    buildingType = SelectField('Type of House', choices = building_type)
    bedrooms  = SelectField('Number of Bedrooms', choices=bedrooms_choice)
    bathrooms= SelectField('Number of Bathrooms', choices=bathrooms_choice)
    zipcode = SelectField("Zipcode", choices=zipcode_choice)
    Squarefeet = FloatField("Squarefeet", validators = [DataRequired()])
    # zipcode = FloatField("Zipcode", validators = [DataRequired()])
    lotsize = FloatField("Lot Size", validators = [DataRequired()])
    city = StringField("City", validators = [DataRequired()])
    yearBuilt = FloatField("Built Year", validators = [DataRequired()])

    # per_sqrft = FloatField("$ per Squarefeet", validators = [DataRequired()])

property_type_choices = [('sfr', 'Single Family Residence'),
                         ('condo', 'Condominium'),
                         ('thr', 'Townhome Residence'),
                         ('mfr', 'Multifamily Residence')
                         ]

def train_flask():
    infile1=open('irvine_data.pk1','rb')
    irvine_train=pickle.load(infile1)

    infile2=open('tustin_data.pk1','rb')
    tustin_train=pickle.load(infile2)

    infile3=open('costamesa_data.pkl','rb')
    costamesa_train=pickle.load(infile3)

    # infile4=open('lakeforest_data.pkl','rb')
    # lakeforest_train=pickle.load(infile4)
    
    cols=['zip','train_built','type','beds','baths','sqrft','lot']
    irvine_df=irvine_train[cols]
    tustin_df=tustin_train[cols]
    costamesa_df=costamesa_train[cols]
    # lakeforest_df=lakeforest_train[cols]
    
    # train['price']=np.log1p(train['price'])
    # train['$/sqrft']=np.log1p(train['$/sqrft'])
    # train['sqrft']=np.log1p(train['sqrft'])
    # train['lot']=np.log1p(train['lot'])

    irvine_size=pd.get_dummies(irvine_df,columns=['zip','type','train_built'])
    tustin_size=pd.get_dummies(tustin_df,columns=['zip','type','train_built'])
    costamesa_size=pd.get_dummies(costamesa_df,columns=['zip','type','train_built'])
    # lakeforest_size=pd.get_dummies(lakeforest_df,columns=['zip','type','train_built'])


    return [irvine_df,irvine_size,tustin_df,tustin_size,costamesa_df,costamesa_size]

def min_built():
    infile1=open('irvine_data.pk1','rb')
    irvine_train=pickle.load(infile1)

    infile2=open('tustin_data.pk1','rb')
    tustin_train=pickle.load(infile2)

    infile3=open('costamesa_data.pkl','rb')
    costamesa_train=pickle.load(infile3)

    # infile4=open('lakeforest_data.pkl','rb')
    # lakeforest_train=pickle.load(infile4)
    
    irvine_mini=irvine_train['built'].min()
    tustin_mini=tustin_train['built'].min()
    costamesa_mini=costamesa_train['built'].min()
    # lakeforest_mini=lakeforest_train['built'].min()


    return [irvine_mini,tustin_mini,costamesa_mini]







n=train_flask()
print(n[0])
print(n[0].columns)

moo=min_built()
print(moo)



