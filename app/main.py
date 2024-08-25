###

import streamlit as st
from streamlit_shap import st_shap

import pickle
import operator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import sklearn
from sklearn.preprocessing import OrdinalEncoder

###

ANN = pickle.load(open('model/model.pkl', 'rb'))
scaler = pickle.load(open('model/scaler.pkl', 'rb'))

X_test_ = pickle.load(open('model/X_test.pkl', 'rb'))
explainer = shap.Explainer(ANN.predict, X_test_)

###

def recode(Age, Sex, Tumor_size, T_category, SRCC, Grade, Location, Histology):
    
    Age_ = Age
    Sex_ = 0 if Sex == 'Female' else 1
    Tumor_size_ = Tumor_size
    T_category_broad_ = {'T1a': 0, 'T1b': 0, 'T2': 1, 'T3': 2, 'T4a': 3, 'T4b': 3}[T_category]
    T_category_ = {'T1a': 0, 'T1b': 1, 'T2': 2, 'T3': 3, 'T4a': 4, 'T4b': 5}[T_category]
    SRCC_ = 0 if SRCC == 'No' else 1
    Grade_ = {'G1': 0, 'G2': 1, 'G3': 2}[Grade]
    
    Location_Lower = True if Location == 'Lower' else False
    Location_Middle = True if Location == 'Middle' else False
    Location_Upper = True if Location == 'Upper' else False
    
    Histology_Diffuse_type = True if Histology == 'Diffuse type' else False
    Histology_Intestinal_type = True if Histology == 'Intestinal type' else False
    Histology_Mixed_Other = True if Histology == 'Mixed/Other' else False

    X_test = pd.DataFrame(
        [Age_, Sex_, Tumor_size_, T_category_broad_, T_category_, SRCC_, Grade_,
        Location_Lower, Location_Middle, Location_Upper,
        Histology_Diffuse_type, Histology_Intestinal_type, Histology_Mixed_Other]
    ).transpose()
    X_test.columns = [
        'Age', 'Sex', 'Tumor size', 'T category, broad', 'T category', 'SRCC', 'Grade', 
        'Location_Lower', 'Location_Middle', 'Location_Upper',
        'Histology_Diffuse type', 'Histology_Intestinal type', 'Histology_Mixed/Other'
    ]
    return X_test

def predict(X_test):
    X_test_scale = scaler.transform(X_test)
    X_test_scale = pd.DataFrame(X_test_scale, columns = X_test.columns)             
    
    X_test_final = X_test_scale[
        ['Tumor size', 'T category, broad', 'T category', 
        'SRCC', 'Grade', 
        'Location_Lower', 'Location_Middle', 'Location_Upper',
        'Histology_Diffuse type', 'Histology_Intestinal type']
    ]
    
    Probability = ANN.predict_proba(X_test_final)[0][1]
    Predicted = ANN.predict(X_test_final)
    st.write(f'Probability of LNM: {Probability*100:.1f}%.')
  
    output = ':red[**Positive**]' if Predicted == True else ':blue[**Negative**]'
    return X_test_final, output

####
def main():
    
    st.set_page_config(layout="wide")
    st.title('GC-LNM')
    col1, col2 = st.columns(2)
    
    with col1:
        
        st.write('An ANN-based prediction model to predict and estimate the probability of lymph node metastasis in patients with gastric cancer.\
            GC-LNM was trained on 500 patients from a Chinese tertiary medical center and validated externally using 824 Asian American patients from Surveillance, Epidemiology, and End Results (SEER) database.\
                This prediction model has been developed and validated solely for scientific research purposes and has not been evaluated prospectively or approved for clinical use.')
    
        st.divider()
        col1_, col2_ = st.columns(2)
        
        with col1_:
            
            Age = st.slider('**Age (years)**',
                            min_value = 18, 
                            max_value = 89)

            Sex = st.radio("**Sex**",
                           ['Female', 'Male'])
            
            Tumor_size = st.slider('**Tumor size (mm)**',
                                   min_value = 1, 
                                   max_value = 300)
        
            T_category = st.radio("**T category (AJCC for GC)**",
                                  ['T1a','T1b','T2','T3','T4a','T4b'])
        
        with col2_:
                
            SRCC = st.radio("**SRCC**",
                             ['No','Yes'])
    
               
            Grade = st.radio("**Grade**",
                             ['G1','G2','G3'])
            
            Location = st.radio("**Location**",
                                ['Lower','Middle','Upper'])
            
            Histology = st.radio("**Histology**",
                                 ['Diffuse type','Intestinal type','Mixed/Other'])

    ####
        
            if "disabled" not in st.session_state:
                st.session_state['disabled'] = False
    
            st.checkbox('**I understand GC-LNM is solely for scientific research purposes.**',
                        key="disabled")
        
            if st.button("**Predict**",
                         disabled=operator.not_(st.session_state.disabled)):
                
                X_test = recode(Age, Sex, Tumor_size, T_category, SRCC, Grade, Location, Histology)
                X_test_final, output = predict(X_test)
                st.success('Prediceted LNM:  {}'.format(output))
####            
    with col2:
                Histology_it = 'Yes' if Histology == 'Intestinal type' else 'No'
                Histology_dt = 'Yes' if Histology == 'Diffuse type' else 'No'
                Location_l =  'Yes' if Location == 'Lower' else 'No'
                Location_m =  'Yes' if Location == 'Middle' else 'No'
                Location_u =  'Yes' if Location == 'Upper' else 'No'
                
                T_category_broad = X_test['T category, broad']
                
                ylabels = [
                    str(Tumor_size) + ' mm' + ' = ' + 'Tumor size',
                    str(T_category_broad) + ' = ' + 'T category, broad',
                    str(T_category) + ' = ' + 'T category',
                    str(SRCC) + ' = ' + 'SRCC', 
                    str(Grade) + ' = ' + 'Grade',
                    str(Location_l) + ' = ' + 'Location - Lower', 
                    str(Location_m) + ' = ' + 'Location - Middle', 
                    str(Location_u) + ' = ' + 'Location - Upper', 
                    str(Histology_dt) + ' = ' + 'Histology - Diffuse type', 
                    str(Histology_it) + ' = ' + 'Histology - Intestinal type',     
                ]

        
                explanation = explainer(X_test_final)

                combine_list = list(zip(
                    np.abs(explanation[0].values),
                    explanation[0].feature_names,
                    ylabels))
                
                sorted_lists = sorted(combine_list, key = lambda x: x[0], reverse = False)
                sorted_ylabels = [item[2] for item in sorted_lists]
                
                st.write('SHAP plot')
                figure = shap.plots.waterfall(explanation[0], max_display=X_test.shape[0], show = False)
                ax_ = figure.get_axes()[0]
                
                tick_labels = ax_.yaxis.get_majorticklabels()
                # for i in range(len(sorted_ylabels)):
                #     tick_labels[i].set_color("black")
                    
                ax_.set_yticks(np.arange(len(sorted_ylabels)))
                ax_.set_yticklabels(sorted_ylabels)
                figure = ax_.get_figure()
                
                st_shap(figure, width=750, height=500)

if __name__=='__main__':
    main()
