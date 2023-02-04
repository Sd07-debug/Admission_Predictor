import streamlit as st
import pandas as pd
from math import sqrt


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,accuracy_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from math import sqrt
from PIL import Image
import webbrowser


admission_df=pd.read_csv('dataset/Admission_Predict.csv')
admission_df.drop('Serial No.',axis=1,inplace=True)
admission_df.rename(columns = {'Chance of Admit ':'Chance of Admit', 'LOR ':'LOR'}, inplace=True)

#Train - Test Split

x = admission_df.drop(columns=['Chance of Admit'])
y = admission_df['Chance of Admit']
x=np.array(x)
y=np.array(y)
y=y.reshape(-1,1)

scaler_x=  StandardScaler()
x=scaler_x.fit_transform(x)
scaler_y=  StandardScaler()
y=scaler_y.fit_transform(y)

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.15)

def LinearReg(data):
    #Linear Regression - Model Loading
    LinearRegression_model=LinearRegression()
    LinearRegression_model.fit(X_train,y_train)

    # Linear Regression - Accuracy
    accuracy_Linear=LinearRegression_model.score(X_test,y_test)


    new_data=[[335,110,1,4.5,4.9,9.6,1],[270,100,5,4.5,4.0,7.6,1],[310,110,3,3.5,5,8.7,1]]
    new_data.append(data)

    fitter=StandardScaler()
    new_data=fitter.fit_transform(new_data)

    res=[]
    for a in new_data:
        b = LinearRegression_model.predict([a])
        res.append(scaler_y.inverse_transform(b))
    
    return [accuracy_Linear,res]

def DecisionTree():
    DecisionTree_model=DecisionTreeRegressor()
    DecisionTree_model.fit(X_train,y_train)
    y_predictd=DecisionTree_model.predict(X_test)

    y_predict_origd=scaler_y.inverse_transform(y_predictd)
    y_test_origd=scaler_y.inverse_transform(y_test)

    MSEd = mean_squared_error(y_test_origd, y_predict_origd)
    r2d = r2_score(y_test_origd, y_predict_origd)

    return [MSEd,r2d,[y_test_origd,y_predict_origd]]

def RandomForest():
    RandomForest_model = RandomForestRegressor(n_estimators=100,max_depth=10)
    RandomForest_model.fit(X_train,np.ravel(y_train))
    y_predictr=RandomForest_model.predict(X_test)

    y_predict_origr=scaler_y.inverse_transform(y_predictr)
    y_test_origr=scaler_y.inverse_transform(y_test)

    MSEr = mean_squared_error(y_test_origr, y_predict_origr)
    r2r = r2_score(y_test_origr, y_predict_origr)

    return [MSEr,r2r,[y_test_origr,y_predict_origr]]

def LinearReg_wd():
    LinearRegression_model=LinearRegression()
    LinearRegression_model.fit(X_train,y_train)
    y_predict=LinearRegression_model.predict(X_test)

    y_predict_orig=scaler_y.inverse_transform(y_predict)
    y_test_orig=scaler_y.inverse_transform(y_test)

    MSEl = mean_squared_error(y_test_orig, y_predict_orig)
    r2l = r2_score(y_test_orig, y_predict_orig)

    return [MSEl,r2l,[y_test_orig,y_predict_orig]]

if __name__ == '__main__':

    st.sidebar.title("Graduate Admission Prediction (GAP)")
    menu = st.sidebar.radio('Navigation',('Introduction',"Data Analysis","Predictor","Model Comparison"))
    st.sidebar.title("Note")
    st.sidebar.info("This Project is based on my paper - 'A Comparative Study of Regression Algorithms for Predicting Graduate Admission to a University'")
    st.sidebar.info("[Read it](https://www.academia.edu/43735168/A_Comparative_Study_of_Regression_Algorithms_for_Predicting_Graduate_Admission_to_a_University?source=swp_share)")
    st.sidebar.info("[Github](https://github.com/ayanbag/Graduate_Admission_Prediction)")
    
    #if st.sidebar.button("View the Paper"):
    #webbrowser.open_new_tab('https://www.academia.edu/43735168/A_Comparative_Study_of_Regression_Algorithms_for_Predicting_Graduate_Admission_to_a_University?source=swp_share')
    #if st.sidebar.button("View Code in Github"):
    #webbrowser.open_new_tab('https://github.com/ayanbag/Graduate_Admission_Prediction')
        
 

    if menu == 'Introduction':
        st.write("""
        # Graduate Admission Prediction
        """)

        st.write("""
        ### Introduction
        """)

        st.markdown("The admission to a graduate program is a very exhaustive and hectic task that requires thorough months of preparation to build a good profile and choosing universities that offer relevant programs. Many of students have problems of knowing their chances of admissions for specific universities based on their resume, and sometimes they will spend plenty of time and energy searching and applying to many academic institutions without getting good results. Some of them are whether going to give up on their plans, get to a lower ranked university, or even face some mental issues such as depression and feeling of failure. Here, using a machine learning approach, we will train our data on a range of parameters, from stellar to mediocre profile. After training, new data will be fed to the system to determine the outcome, and sample data will be tested , in order to understand the performance of our implemented model both in theoretically and visually. In this project, all parameters necessary for the admission purpose like GRE Score, TOEFL Score, University Rating, Statement of Purpose and Letter of Recommendation Strength, Undergraduate GPA, and Research Experience are taken into consideration.")

        st.write("""
        ### Dataset
        """)

        st.dataframe(admission_df)

        st.markdown("The dataset, we are using, consists of 400 grad student’s records containing information about the GRE Scores (in 340 scale), TOEFL Scores (in 120 scale), University Rating (on scale of 5), Statement of Purpose Strength (on scale of 5), Letter of Recommendation Strength (on scale of 5), CGPA (Undergraduate Grade Point), Research Experience (yes or no in which 1 denotes Yes and 0 denotes No) and Chance of Admit (a value between 0 and 1), being the target variable. In the data-set, 6 features are continuous with only 1 feature, Research Experience as categorical.")

        st.write("""
        #### More information about our dataset
        """)
        st.write(admission_df.describe())


    if menu == 'Data Analysis':
        st.write("""
        # Exploratory Data Analysis of the Dataset
        """)

        import plotly.express as px

        #fig1, ax = plt.subplots(figsize=(10,10))
        fig, ax = plt.subplots(figsize=(10,10))
        sns.heatmap(admission_df.corr(), annot=True, cmap='Spectral')

        st.pyplot()

        st.markdown("""
        The features that affect the "Chance to Admit" are: (Importance is in Descending Order)

        - CGPA (0.87)
        - GRE Score (0.80)
        - TOEFL Score (0.79)
        - University Rating (0.71)
        - Statement of Purpose (SOP) (0.68)
        - Letter of Recommendation (LOR) (0.67)
        - Research Experience (0.55)
        """
        )

        st.write("Let's explore these features to get a better understanding")

        select_gr= st.selectbox('Select any Feature', ['CGPA','GRE Score',"TOEFL Score","University Rating","Research Experience"])

        if select_gr == 'CGPA':

            st.markdown("""  """)
            st.write("""
            #### CGPA vs Chance of Admit
            """)
            st.markdown("""  """)

            st.markdown("""
            The Cumulative Grade Point Average is a 10 point grading system.
            From the data shown below, it appears the submissions are normally distributed. With a mean of 8.6 and standard deviation of 0.6.
            


            **Moreover, it appears as applicant's CGPA has a strong correlation with their Chance of Admission.**
            """)


            plt.figure(figsize=(20,8))
            plt.subplot(1,2,1)
            sns.distplot(admission_df['CGPA'],color = 'r')
            plt.title('CGPA Distribution of Applicants')
            plt.subplot(1,2,2)
            sns.regplot(admission_df['CGPA'], admission_df['Chance of Admit'],color = 'g')
            plt.title('CGPA vs Chance of Admit')

            st.pyplot()
        
        elif select_gr == 'GRE Score':

            
            st.markdown("""  """)
            st.write("""
            #### GRE Score vs Chance of Admit
            """)
            st.markdown("""  """)

            
            st.markdown("""
            The Graduate Record Examination is a standarized exam, often required for admission to graduate and MBA programs globally. It's made up of three components:

            - Analytical Writing (Scored on a 0-6 scale in half-point increments)
            - Verbal Reasoning (Scored on a 130-170 scale)
            - Quantitative Reasoning (Scored on a 130-170 scale)


            In this dataset, the GRE Score is based on a maximum of 340 points. The mean is 317 with a standard deviation of 11.5.
            
            
            
            

            **Moreover, it appears as applicant's GRE Score has a strong correlation with their Chance of Admission, but less than that of CGPA.**
            """)

            plt.figure(figsize=(20,8))
            plt.subplot(1,2,1)
            sns.distplot(admission_df['GRE Score'],color='m')
            plt.title('Distributed GRE Score of Applicants')

            plt.subplot(1,2,2)
            sns.regplot(admission_df['GRE Score'], admission_df['Chance of Admit'],color = 'b')
            plt.title('GRE Scores vs Chance of Admit')

            st.pyplot()

        elif select_gr == "TOEFL Score":

            st.markdown("""  """)
            st.write("""
            #### TOEFL Score vs Chance of Admit
            """)
            st.markdown("""  """)

            st.markdown(
            """
            The Test of English as a Foreign Language is a standarized test for non-native English speakers that are choosing to enroll in English-speaking universities.

            The test is split up into 4 sections:

            - Reading
            - Listening
            - Speaking
            - Writing


            All sections are scored out of 30, giving the exam a total score of 120 marks. In this dataset, the TOEFL scores have a mean of 107 and a standard deviation of 6.

            """
            )

            plt.figure(figsize=(20,8))
            plt.subplot(1,2,1)
            sns.distplot(admission_df['TOEFL Score'],color='g')
            plt.title('Distributed TOEFL Scores of Applicants')

            plt.subplot(1,2,2)
            sns.regplot(admission_df['TOEFL Score'], admission_df['Chance of Admit'],color = 'c')
            plt.title('TOEFL Scores vs Chance of Admit')

            st.pyplot()

        elif select_gr == "University Rating":

            st.markdown("""  """)
            st.write("""
            #### University Rating vs Number of Applicants
            """)
            st.markdown("""  """)

            st.markdown(""" 
            
            The data analysis shows that the most applicants come from a 3 Star and 2 Star university.
            
            """)

            fig, ax = plt.subplots(figsize=(8,6))
            sns.countplot(admission_df['University Rating'])
            plt.title('University Rating')
            plt.ylabel('Number of Applicants')

            st.pyplot()
        

        elif select_gr == "Research Experience":
            st.markdown("""  """)
            st.write("""
            #### Research Experience vs Number of Applicants
            """)
            st.markdown("""  """)

            st.markdown("""
            It seems the majority of applicants have research experience. However, this is the least important feature, so it doesn't matter all too much if an applicant has the experience or not.
            """)

            
            fig, ax = plt.subplots(figsize=(8,6))
            sns.countplot(admission_df['Research'])
            plt.title('Research Experience')
            plt.ylabel('Number of Applicants')
            ax.set_xticklabels(['No Research Experience', 'Has Research Experience'])

            st.pyplot()

    if menu == 'Predictor':
        st.write("""
        # Prediction using Machine Learning Algorithm
        """)

        st.markdown("""
        I have implemented Linear Regression Algorithm to predict the chance of admission to a Graduate University, as it is found to the best algorithm that fits perfectly with the dataset.


        To know more about it, check out my paper
        """)
       
        st.write("""
        #### Fill the following data
        """)

        #name = st.text_input("Name","")
        gre = st.number_input("Enter your GRE Score")
        toefl = st.number_input("Enter Your TOEFL Score")
        ur = st.slider("Enter Your University Ratings",1,5,1)
        sop = st.slider("Enter Your Statement of Purpose Strength",1.0,5.0,1.0)
        lor = st.slider("Enter Your Letter of Recomendation Strength",1.0,5.0,1.0)
        cgpa = st.number_input("Enter Your CGPA (out of 10)")
        rexp = st.selectbox("Has Research Experience",["Yes","No"])

        if rexp == 'Yes':
            rxp = 1
        else: 
            rxp = 0
        if gre < 0 or gre >340:
            st.error("GRE Score should in the range of 0 - 340")
            st.warning("This may lead to wrong prediction")

        if cgpa < 0 or cgpa >10.00:
            st.error("CGPA should in the range of 0 - 10.00")
            st.warning("This may lead to wrong prediction")

        if toefl < 0 or toefl >120.00:
            st.error("TOEFL Score should in the range of 0 - 120")
            st.warning("This may lead to wrong prediction")
        
        if (toefl <=120 and toefl>=0)  and (gre <=340 and gre>=0) and (cgpa <=10.00 and cgpa>=0):
            
            nd = [gre,toefl,ur,sop,lor,cgpa,rxp]
            if st.button("Predict"):
                acc,result = LinearReg(nd)
                
                import time

                my_bar = st.progress(0)

                for percent_complete in range(100):
                    time.sleep(0.1)
                    my_bar.progress(percent_complete + 1)


                st.write("""
                ### Model Accuracy : 
                """)
                st.write(str(acc*100)+"%")


                st.write("""
                ### Predicted Chance of Admission based on the current profile :
                """)

                val = int(result[-1][0][0]*100)

                if val >=70 and val <=100:
                    st.success(str(val)+ "%")
                elif val < 70 and val >= 40:
                    st.warning(str(val)+ "%")
                else:
                    st.error(str(val)+ "%")

    if menu == "Model Comparison":

        st.write("""
        # Comparison of Linear Regression Model with Other Regression Model
        """)

        st.markdown(
        """
        We will be comparing our **Linear Regression Model** with  :

        - Decision Tree Regression Model
        - Random Forest Regression Model
        """
        )

        st.markdown("""  """)
        
        st.write("""
        #### Select the Model to compare with our implemented model
        """)

        msel,r2l,grpl = LinearReg_wd()

        select_model= st.selectbox('', ["Random Forest Regression","Decision Tree Regression","Both"])


        if select_model == "Random Forest Regression":

            mser,r2r,grpr=RandomForest()

            data1= { 'Models': ['Linear Regression','Random Forest Regression'],
                    'R-Square Score': [r2l,r2r],
                    'Mean Sqaure Error': [msel,mser],
                    'Accuracy (in %)': [int(r2l*100),int(r2r*100)]
            }
            tabl = pd.DataFrame(data1,columns=['Models','R-Square Score','Mean Sqaure Error','Accuracy (in %)'])

            st.table(tabl)

            plt.figure(figsize=(20,7))
            plt.subplot(1,2,1)
            plt.plot(grpl[0],grpl[1],"^",color='r')
            plt.ylabel("Predicted Chance of Admission")
            plt.xlabel("Actual Chance of Admission")
            plt.title('Predicted vs Actual Values(Linear Regression)')

            plt.subplot(1,2,2)

            plt.plot(grpr[0],grpr[1],"^",color='m')
            plt.ylabel("Predicted Chance of Admission")
            plt.xlabel("Actual Chance of Admission")
            plt.title('Predicted vs Actual Values(Random Forest Regression)')

            st.pyplot()

        elif select_model == "Decision Tree Regression":
    
            msed,r2d,grpd=DecisionTree()

            data2= { 'Models': ['Linear Regression','Decision Tree Regression'],
                    'R-Square Score': [r2l,r2d],
                    'Mean Sqaure Error': [msel,msed],
                    'Accuracy (in %)': [int(r2l*100),int(r2d*100)]
            }
            tab2 = pd.DataFrame(data2,columns=['Models','R-Square Score','Mean Sqaure Error','Accuracy (in %)'])

            st.table(tab2)

            plt.figure(figsize=(20,7))
            plt.subplot(1,2,1)
            plt.plot(grpl[0],grpl[1],"^",color='g')
            plt.ylabel("Predicted Chance of Admission")
            plt.xlabel("Actual Chance of Admission")
            plt.title('Predicted vs Actual Values(Linear Regression)')

            plt.subplot(1,2,2)

            plt.plot(grpd[0],grpd[1],"^",color='b')
            plt.ylabel("Predicted Chance of Admission")
            plt.xlabel("Actual Chance of Admission")
            plt.title('Predicted vs Actual Values(Decision Tree Regression)')

            st.pyplot()

        elif select_model == "Both":
        
            msed,r2d,grpd=DecisionTree()
            mser,r2r,grpr=RandomForest()


            data3= { 'Models': ['Linear Regression','Random Forest Regression','Decision Tree Regression'],
                    'R-Square Score': [r2l,r2r,r2d],
                    'Mean Sqaure Error': [msel,mser,msed],
                    'Accuracy (in %)': [int(r2l*100),int(r2r*100),int(r2d*100)]
            }
            tab3 = pd.DataFrame(data3,columns=['Models','R-Square Score','Mean Sqaure Error','Accuracy (in %)'])

            st.table(tab3)

            plt.figure(figsize=(20,7))
            plt.subplot(1,2,1)
            plt.plot(grpl[0],grpl[1],"o",color='c')
            plt.ylabel("Predicted Chance of Admission")
            plt.xlabel("Actual Chance of Admission")
            plt.title('Predicted vs Actual Values(Linear Regression)')

            plt.subplot(1,2,2)

            plt.plot(grpr[0],grpr[1],"o",color='r')
            plt.ylabel("Predicted Chance of Admission")
            plt.xlabel("Actual Chance of Admission")
            plt.title('Predicted vs Actual Values(Random Forest Regression)')

            st.pyplot()

            plt.figure(figsize=(20,7))
            plt.subplot(1,2,1)
            plt.plot(grpd[0],grpd[1],"o",color='m')
            plt.ylabel("Predicted Chance of Admission")
            plt.xlabel("Actual Chance of Admission")
            plt.title('Predicted vs Actual Values(Decision Tree Regression)')
            st.pyplot()

        
