# Importing neccesary Libraries 
import streamlit as st 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA 

#Over Sampling and under sampling libraries
##from imblearn.over_sampling import SMOTE
##from imblearn.under_sampling import RandomUnderSampler

import warnings
warnings.filterwarnings("ignore")


def main():
    st.title("Project:- Credit Card fraud Detection")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Credit Card fraud Detector </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)


    st.subheader("Performing following Activities: ")
    st.subheader("1.EDA 2.Data Visualization 3.Artificial Neural Network")
 

    activity=["Data analysis","Data Visualization","Fraud Detection"]
    selectact= st.sidebar.selectbox("Choose Analysis Method",activity)
    if(selectact=="Data analysis"):
        if (st.button("Project Details")):
            st.write("""We are going to build project on The Deep Learning model to detect the Credit Card fraud done by European cardholders in september 2013.
                We will learn Deep Learning concepts such as Artification Neural Networks,Layers,Activation Functions,Neurons.
                and top of that we will build Deep Learning Algorithm which mimics humain br ain
                We analyse data to get meaning information from them and Visualize them to recognise different patterns among them and showing meaning full information
                It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase.""")

        if (st.button("Data set Details")):
            st.write("""
        The dataset contains transactions made by credit cards in September 2013 by European cardholders.
        This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions.
        The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
        It contains only numerical input variables which are the result of a PCA transformation.
        Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data.
        Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'.
        Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset.
        The feature 'Amount' is the transaction Amount,
        """)
        st.subheader("Exploratory Data Analysis")
        df = pd.read_csv("creditcard.csv")
        if(st.checkbox("Preview Dataset")):
            number = st.number_input("Select Number of Rows to View", value=1)
            st.write(df.head(number))
        if(st.checkbox("Shape of Dataset")):
            st.write(df.shape)
            data_dim = st.radio("Show Dimension By ", ("Rows","Columns","Size"))
            if(data_dim == "Rows"):
                st.text("Showing the Rows")
                st.write(df.shape[0])
            if(data_dim=="Columns"):
                st.text("Showing the Columns")
                st.write(df.shape[1])
            if(data_dim=="Size"):
                st.text("Showing the Size")
                st.write(df.size)
                
        if(st.checkbox("Select Columns")):
            all_columns = df.columns
            selected_columns = st.multiselect("Select Columns",all_columns)
            new_df = df[selected_columns]
            st.dataframe(new_df)
            
        if(st.button("Descriptive summary")):
            st.write(df.describe())

        if(st.button("Info summary")):
            info=""" RangeIndex: 284807 entries, 0 to 284806
Data columns (total 31 columns):
 #   Column  Non-Null Count   Dtype  
---  ------  --------------   -----  
 0   Time    284807 non-null  float64
 1   V1      284807 non-null  float64
 2   V2      284807 non-null  float64
 3   V3      284807 non-null  float64
 4   V4      284807 non-null  float64
 5   V5      284807 non-null  float64
 6   V6      284807 non-null  float64
 7   V7      284807 non-null  float64
 8   V8      284807 non-null  float64
 9   V9      284807 non-null  float64
 10  V10     284807 non-null  float64
 11  V11     284807 non-null  float64
 12  V12     284807 non-null  float64
 13  V13     284807 non-null  float64
 14  V14     284807 non-null  float64
 15  V15     284807 non-null  float64
 16  V16     284807 non-null  float64
 17  V17     284807 non-null  float64
 18  V18     284807 non-null  float64
 19  V19     284807 non-null  float64
 20  V20     284807 non-null  float64
 21  V21     284807 non-null  float64
 22  V22     284807 non-null  float64
 23  V23     284807 non-null  float64
 24  V24     284807 non-null  float64
 25  V25     284807 non-null  float64
 26  V26     284807 non-null  float64
 27  V27     284807 non-null  float64
 28  V28     284807 non-null  float64
 29  Amount  284807 non-null  float64
 30  Class   284807 non-null  int64  
dtypes: float64(30), int64(1)
memory usage: 67.4 MB"""
            st.text("info of data set:{}".format(info))
        if(st.button("Target Class Distribution")):
            st.write(df["Class"].value_counts())
            fig = plt.figure()
            df['Class'].value_counts().plot(kind='bar')
            plt.show()
            st.pyplot(fig)
            
            
        if(st.button("Under Sampling")):
            fraud_df = df.loc[df['Class'] == 1]
            non_fraud_df = df.loc[df['Class'] == 0].sample(n=492)
            df_new = pd.concat([fraud_df, non_fraud_df])
            st.write(df_new["Class"].value_counts())


##    if(selectact=="Data Visualization"):
##        st.subheader("Data Visualizaton")
##        df = pd.read_csv("vgsales.csv")
##
##        if(st.checkbox("Overall visualization")):
##            st.subheader("Using Pairplot Graph")
##            fig = plt.figure()
##            sns.pairplot(df)
##            plt.show()
##            st.pyplot(fig)
##
##        if(st.sidebar.checkbox("Correlation Visualization")):
##            st.subheader("Correlation between independent variable and target variable(Global Sales)")
##            fig = plt.figure()
##            sns.heatmap(df.corr(),annot=True,cmap='RdYlBu_r')
##            plt.show()
##            st.pyplot(fig)
##
##        if(st.sidebar.checkbox("genrewise Sales Comparison")):
##            st.subheader("Genrewise Regions Sales Comparison Analysis")
##            df1 = df[['Genre', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']]
##            df1 = df1.groupby("Genre").sum()
##            fig = plt.figure()
##            sns.heatmap(df1, annot=True,fmt=".2f")
##            plt.show()
##            st.pyplot(fig)
##
##        if(st.sidebar.checkbox("Global sales data")):
##            fig = plt.figure()
##            plt.hist(df["Global_Sales"],bins=20)
##            plt.xticks(np.arange(0,80,5))
##            plt.xlabel("Sale in million")
##            plt.show()
##            st.pyplot(fig)
##
##        if(st.sidebar.checkbox("Yearly analysis")):
##            fig = plt.figure()
##            plt.hist(df["Year"],bins=40)
##            plt.xticks(rotation=90)
##            plt.ylabel("Games released")
##            plt.show()
##            st.pyplot(fig)
##
##        if(st.sidebar.checkbox("Region wise Global Sales")):
##            st.subheader("Region wise Global Sales analysis")
##            chart_type=st.sidebar.selectbox("Select chart type: ",["Pie Chart","Bar Graph"])
##            if(chart_type=="Pie Chart"):
##                fig = plt.figure()
##                df2= df[["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]]
##                df2= df2.sum().reset_index()
##                plt.pie(df2[0],labels=df2["index"], autopct='%1.2f%%')
##                plt.show()
##                st.pyplot(fig)
##            else:
##                fig = plt.figure()
##                df2= df[["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]]
##                df2= df2.sum().reset_index()
##                plt.bar(df2["index"],df2[0])
##                plt.ylabel("Global Sales")
##                plt.xlabel("Regions")
##                plt.show()
##                st.pyplot(fig)
##
##        if(st.sidebar.checkbox("Genre wise Global Sales")):
##            st.subheader("Genre wise Global Sales analysis")
##            chart_type=st.sidebar.selectbox("Select chart type: ",["Pie Chart","Bar Graph"])
##            if(chart_type=="Pie Chart"):
##                fig = plt.figure()
##                x_val=df["Genre"].unique()
##                y_val=df.groupby("Genre")["Global_Sales"].sum()
##                plt.pie(y_val,labels=x_val, autopct='%1.2f%%')
##                plt.show()
##                st.pyplot(fig)
##            else:
##                fig = plt.figure()
##                x_val=df["Genre"].unique()
##                y_val=df.groupby("Genre")["Global_Sales"].sum()
##                plt.bar(x_val,y_val,color="maroon")
##                plt.xticks(rotation=90)
##                plt.xlabel("Genre's type")
##                plt.ylabel("Global Sale")
##                plt.show()
##                st.pyplot(fig)
##                
##        if(st.sidebar.checkbox("Year wise Global Sales")):
##            st.subheader("Year wise Global Sales analysis")
##            fig = plt.figure()
##            dfyear = df.groupby(['Year'])['Global_Sales'].sum()
##            dfyear = dfyear.reset_index()
##            sns.barplot(x="Year", y="Global_Sales", data=dfyear)
##            plt.xticks(rotation=90)
##            plt.ylabel("Global Sale")
##            plt.grid(True)
##            plt.show()
##            st.pyplot(fig)
##
##        if(st.sidebar.checkbox("Platfrom wise Global Sales")):
##            st.subheader("Platform wise Global Sales analysis")
##            fig = plt.figure()
##            dfyear = df.groupby(['Platform'])['Global_Sales'].sum().reset_index().sort_values("Global_Sales",ascending=False) 
##            dfyear = dfyear.reset_index()
##            sns.barplot(x="Platform", y="Global_Sales", data=dfyear)
##            plt.xticks(rotation=90,fontsize=12)
##            plt.xlabel("Platform",fontsize=12)
##            plt.ylabel("Global Sale",fontsize=12)
##            plt.grid(True)
##            plt.show()
##            st.pyplot(fig)
##
##        if(st.sidebar.checkbox("Year wise games released")):
##            st.subheader("Year wise games released analysis")
##            fig = plt.figure()
##            yrgame = df.groupby('Year')['Name'].count()
##            sns.countplot(x="Year", data=df, order=yrgame.index)
##            plt.xticks(rotation=90)
##            plt.ylabel("No of games released")
##            plt.grid(True)
##            plt.show()
##            st.pyplot(fig)
##
##        if(st.sidebar.checkbox("Genre wise Games Released")):
##            st.subheader("Genre wise Games released analysis")
##            fig = plt.figure()
##            sns.countplot(x="Genre", data=df, order = df['Genre'].value_counts().index)
##            plt.xticks(rotation=90,fontsize=12)
##            plt.xlabel("Genre",fontsize=12)
##            plt.grid(True)
##            plt.show()
##            st.pyplot(fig)
##
##        if(st.sidebar.checkbox("Platform wise Games Released")):
##            st.subheader("Platform wise Games released analysis")
##            fig = plt.figure()
##            sns.countplot(x="Platform", data=df, order = df['Platform'].value_counts().index)
##            plt.xticks(rotation=90,fontsize=12)
##            plt.grid(True)
##            plt.show()
##            st.pyplot(fig)
##
##                
##            
##        
##           
    if(selectact=="Fraud Detection"):
        st.subheader("Credit Card Fraud detection Deep learning model")
##        time = st.number_input("Enter Time)",0.1,100.0)
##        v1 = st.number_input("Enter v1)",0.1,100.0)
##        v2 = st.number_input("Enter v2)",0.1,100.0)
##        v3 = st.number_input("Enter v3)",0.1,100.0)
##        v4 = st.number_input("Enter v4)",0.1,100.0)
##        v5 = st.number_input("Enter v5)",0.1,100.0)
##        v6 = st.number_input("Enter v6)",0.1,100.0)
##        v7 = st.number_input("Enter v7)",0.1,100.0)
##        v8 = st.number_input("Enter v8)",0.1,100.0)
##        v9 = st.number_input("Enter v9)",0.1,100.0)
##        v10 = st.number_input("Enter v10)",0.1,100.0)
##        v11= st.number_input("Enter v11)",0.1,100.0)
##        v12= st.number_input("Enter v12)",0.1,100.0)
##        v13= st.number_input("Enter v13)",0.1,100.0)
##        v14= st.number_input("Enter v14)",0.1,100.0)
##        v15= st.number_input("Enter v15)",0.1,100.0)
##        v16= st.number_input("Enter v16)",0.1,100.0)
##        v17= st.number_input("Enter v17)",0.1,100.0)
##        v18= st.number_input("Enter v18)",0.1,100.0)
##        v19= st.number_input("Enter v19)",0.1,100.0)
##        v20= st.number_input("Enter v20)",0.1,100.0)
##        v21= st.number_input("Enter v21)",0.1,100.0)
##        v22= st.number_input("Enter v22)",0.1,100.0)
##        v23= st.number_input("Enter v23)",0.1,100.0)
##        v24= st.number_input("Enter v24)",0.1,100.0)
##        v25= st.number_input("Enter v25)",0.1,100.0)
##        v26= st.number_input("Enter v26)",0.1,100.0)
##        v27= st.number_input("Enter v27)",0.1,100.0)
##        v28= st.number_input("Enter v28)",0.1,100.0)
##        amount = st.number_input("Enter Amount)",0.1,100.0)
##        
##
##        result=[[time,v1,v2,v3,v4,v5,v6]]
##        disp_result={"NA_Sales ":na,
##                     "EU_Sales":eu,
##                     "JP_Sales":jp,
##                     "Other_Sales":ot}
##        st.info(result)
##        st.json(disp_result)
##
##        st.subheader("Make Prediction")
##
##        regressor = st.sidebar.selectbox('Select ML model',('Linear Regression', 'KNN regressor', 'Decision Tree','Random Forest','SVR'))
##
##        def add_param(regressor):
##            params = dict()
##            if regressor == 'SVR':
##                C = st.sidebar.slider('C', 0.01, 10.0)
##                params['C'] = C
##                sel_kernel=st.sidebar.radio("Select Kernel",(("linear","rbf")))
##                if(sel_kernel=="linear"):
##                    params["kernel"]=sel_kernel
##                else:
##                    params["kernel"]=sel_kernel
##                    
##            elif regressor == 'Linear Regression':
##                st.sidebar.text("No parameters")
##            elif regressor == 'KNN regressor':
##                K = st.sidebar.slider('K', 1, 15)
##                params['K'] = K
##            elif regressor == 'Decision Tree':
##                max_depth = st.sidebar.slider('max_depth',2,15)
##                params['max_depth'] = max_depth
##                min_sample = st.sidebar.slider('min_samples_leaf',1,15)
##                params['min_samples_leaf'] = min_sample
##            else:
##                max_depth = st.sidebar.slider('max_depth', 2, 15)
##                params['max_depth'] = max_depth
##                n_estimators = st.sidebar.slider('n_estimators', 1, 100)
##                params['n_estimators'] = n_estimators
##            return params
##        params = add_param(regressor)
##
##        def get_regressor(regressor, params):
##            reg = None
##            if regressor == 'SVR':
##                reg = SVR(C=params['C'],kernel=params["kernel"])
##            elif regressor == 'KNN regressor':
##                reg = KNeighborsRegressor(n_neighbors=params['K'])
##            elif regressor == 'Decision Tree':
##                reg = DecisionTreeRegressor(max_depth=params['max_depth'],min_samples_leaf=params['min_samples_leaf'])
##            elif regressor == 'Random Forest':
##                reg = RandomForestRegressor(max_depth=params['max_depth'],n_estimators=params['n_estimators'])
##            else:
##                reg = LinearRegression()
##            return reg
##        reg = get_regressor(regressor, params)
##
##        
        # Model Creation
        df = pd.read_csv("creditcard.csv")
        # Missing Values:
        if(st.sidebar.checkbox("Any Missing Values")):
            st.text("missing values count")
            st.write(df.isna().sum())
            if(st.button("Remove if present")):
                st.text("Drop missing values rows")
                df.dropna(inplace=True)
                st.write(df.isna().sum())
                
        fraud_df = df.loc[df['Class'] == 1]
        non_fraud_df = df.loc[df['Class'] == 0].sample(n=492)
        df_new = pd.concat([fraud_df, non_fraud_df])
        st.write(df_new["Class"].value_counts())
        
        x = df_new.drop("Class",axis=1)
        y = df_new["Class"]
        st.write('X data(head()):', pd.DataFrame(x).head())
        st.write('y data:(head())', pd.DataFrame(y).head())
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
        
        # Scaling
        sc=StandardScaler()
        x_train_ss=sc.fit_transform(x_train)
        x_test_ss=sc.transform(x_test)

        # Neural network
        model=Sequential()
        model.add(Dense(32,activation='relu',input_dim=30))
        model.add(Dense(64,activation='relu'))
        model.add(Dense(64,activation='relu'))
        model.add(Dense(1,activation='sigmoid'))

        model.compile(optimizer="adam",loss="binary_crossentropy")
        trained_model = model.fit(x_train_ss,y_train,epochs=20,batch_size=50)

        y_pred = model.predict(x_test_ss)
        y_pred = np.where(y_pred >= 0.5,1,0)
        
        acc =classification_report(y_test,y_pred)
        st.success("Accuracy of model is{}".format(acc))

        

                
        
if __name__=='__main__':
    main()
    
