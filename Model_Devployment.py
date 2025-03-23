#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pickle
import streamlit as st


# In[3]:


st.title('DIABETES PREDICTION')


# In[4]:


loaded_model= pickle.load(open('Diabetesmodel.pkl','rb'))


# In[6]:


def Disease(input_data):
    input_data_as_numpy_array= np.asarray(input_data)
    input_reshape= input_data_as_numpy_array.reshape(1,-1)

    prediction= loaded_model.predict(input_reshape)

    if(prediction[0]==0):
        return st.success('The person has not Diabetes')
    else:
        return st.error('The person has Diabets')

def main():
    st.write("Pridiction Model")

    BMI= st.number_input('Enter Body Mass Index')
    Insulin= st.number_input('Enter Insulin', step= 2)
    Glucose= st.number_input('Enter Glucose', step= 2)
    Age= st.number_input('Enter your Age', step= 2)

    diagnosis= ''

    if st.button('PREDICT'):
        diagnosis= Disease([Glucose,Insulin,BMI,Age])


if __name__== '__main__':
    main()


# In[ ]:




