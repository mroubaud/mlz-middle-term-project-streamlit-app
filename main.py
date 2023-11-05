import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle 
import sklearn

from sklearn.model_selection import train_test_split

intro = st.container()
exp_data = st.container()
final_model = st.container()
user_int = st.container()

###Section 1: APP presentation and functions

with intro:
  st.title("Predict Salary App")

  st.write("""This App predict the salary (in USD) a employee working in the data science field should have. 
         This prediction is base on many different features, such as:""")

  st.markdown("**Work Year**: The year the salary was paid")

  st.markdown("""**Experience Level**: The experience level in the job during the year
  - *EN*: Entry-level / Junior
  - *MI*: Mid-level / Intermediate
  - *SE*: Senior-level / Expert
  - *EX*: Executive-level / Director""")

  st.markdown("""**Employment Type**: The experience level in the job during the year
  - *PT*: Part-time
  - *FT*: Full-time
  - *CT*: Contract
  - *FL*: Freelance""")

  st.markdown("**Job Title**: The experience level in the job during the year")

  st.markdown("**Employee Residence**: Employee's primary country of residence in during the work year as an ISO 3166 country code")

  st.markdown("""**Remote Ratio**: The overall amount of work done remotely, possible values are as follows:")
  - *0*: No remote work (less than 20%)
  - *50*: Partially remote/hybrid"
  - *100*: Fully remote (more than 80%)""") 

  st.markdown("**Company Location**: The country of the employer's main office or contracting branch as an ISO 3166 country code.")

  st.markdown("""**Company Size**: The average number of people that worked for the company during the year:"
  - *S*: less than 50 employees (small)
  - *M*: 50 to 250 employees (medium) 
  - *L*: more than 250 employees (large)""")

  st.write("""You can find the data set used to train the model and the scripts in this github repo""")

  ## Grouping by Job Title Cryteria:
  st.subheader("Job Titles Considered:")
  st.write("""In our dataset we have 119 different positions, and probably, for some positions we will have just a few samples. 
         Given this, we will group similar positions under the same group. 
         In this way pur model will have much sence and it will be more robust. 
         We are going to differenciate between the following positions:""")
  st.markdown(""" 
  - Data Analyst 
  - Data Engineers 
  - Machine Learning Engineer 
  - Artificial Intelligent Engineer
  - Data Scientist 
  - Data Architect 
  - Business Intelligence Engineer
  - Data and Analytics Manager """)

###Section 2: Exploratory data Analysis base on train set 
df = pd.read_csv("salaries.csv")
df = df.drop(["salary","salary_currency"], axis=1)
categorical_columns = df.select_dtypes(include="object").columns.values
numerical_columns = df.select_dtypes(include="int64").columns.values
for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(" ","_")
#function for classify job_title column according above cryteria
def classify_job_title(job_title):
    if ((job_title == "data_analyst") | (job_title == "staff_data_analyst")
       |(job_title == "data_operations_analyst")|(job_title == "product_data_analyst") 
       |(job_title == "data_visualization_analyst")|(job_title == "marketing_data_analyst")):
        return "data_analyst"
    elif ((job_title == "data_engineer")):
        return "data_engineer"
    elif ((job_title == "machine_learning_engineer") | (job_title == "ml_engineer")
        |(job_title == "machine_learning_scientist") | (job_title == "machine_learning_specialist")
        |(job_title == "machine_learning_researcher") | (job_title == "machine_learning_software_engineer")
        |(job_title == "applied_machine_learning_scientist") | (job_title == "machine_learning_research_engineer") 
        |(job_title == "machine_learning_developer")):
        return "machine_learning_engineer"
    elif ((job_title == "ai_developer") | (job_title == "ai_programmer")
        |(job_title == "ai_engineer") | (job_title == "deep_learning_engineer")
        |(job_title == "computer_vision_software_engineer") | (job_title == "deep_learning_researcher")
        |(job_title == "ai_research_engineer") | (job_title == "ai_scientist")
        |(job_title == "computer_vision_engineer")):
        return "artificial_intelligence_engineer"
    elif ((job_title == "data_scientist") | (job_title == "principal_data_scientist")
        |(job_title == "data_science_engineer")):
        return "data_scientist"
    elif ((job_title == "data_architect") | (job_title == "data_infrastructure_engineer")
        |(job_title == "machine_learning_infrastructure_engineer") | (job_title == "ai_architect")
        |(job_title == "cloud_data_engineer") | (job_title == "aws_data_architect")
        |(job_title == "cloud_database_engineer") | (job_title == "big_data_architect")
        |(job_title == "principal_data_architect") | (job_title == "cloud_data_architect")):
        return "data_architect"
    elif ((job_title == "business_analyst") | (job_title == "business_intelligence_engineer")
        |(job_title == "business_intelligence_analyst") | (job_title == "bi_analyst")
        |(job_title == "business_data_analyst") |(job_title == "bi_data_analyst") 
        |(job_title == "business_intelligence_data_analyst")):
        return "business_intelligence_engineer"
    elif ((job_title == "data_and_analytics_manager") | (job_title == "data_lead") 
        | (job_title == "data_strategy_manager") | (job_title == "data_manager")
        | (job_title == "data_science_manager") | (job_title == "data_scientist_lead")
        | (job_title == "data_analytics_manager") | (job_title == "data_analytics_lead")
        | (job_title == "lead_data_engineer") | (job_title == "data_science_lead")
        | (job_title == "lead_data_scientist")):
        return "data_and_analytics_manager"
    else:
        return "other"
df["job_title"] = df["job_title"].apply(classify_job_title)
df = df[df["job_title"] != "other"]
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_full_train = df_full_train.reset_index().drop("index", axis=1)
#Bar plot job title distribution
job_title_bar_plot = plt.figure(figsize=(10,4),dpi=200)
plt.title("Samples of each job of interest")
sns.countplot(data=df_full_train,x="job_title")
plt.xticks(rotation=90);
#Bar plot salary vs job_title
job_title_box_plot =plt.figure(figsize=(10,4),dpi=200)
sns.boxplot(data=df_full_train,y='salary_in_usd',x='job_title')
plt.title("Box plot Salary vs Job Title")
plt.xticks(rotation=90);
#Box plot salary vs employment_type
employment_type_box_plot=plt.figure(figsize=(10,4),dpi=200)
sns.boxplot(data=df_full_train,y='salary_in_usd',x='employment_type')
plt.title("Box plot Salary vs Employment Type")
plt.xticks(rotation=90);
#Box plot salary vs experience
experience_box_plot=plt.figure(figsize=(10,4),dpi=200)
sns.boxplot(data=df_full_train,y='salary_in_usd',x='experience_level')
plt.title("Box plot Salary vs Experience Level")
#Bar plot experience_level
experience_bar_plot=plt.figure(figsize=(10,4),dpi=200)
plt.title("Bar plot Experience Level")
sns.countplot(data=df_full_train,x="experience_level")
plt.xticks(rotation=90);
#Correlation Matrix between features
df_full_train["work_year"] = df_full_train["work_year"].replace({2020:1, 2021:2, 2022:3, 2023:4})
df_full_train["experience_level"] = df_full_train["experience_level"].replace({'se':1, 'en':2, 'mi':3, 'ex':4})
df_full_train["company_size"] = df_full_train["company_size"].replace({'s':1, 'm':2, 'l':3})
df_full_train["remote_ratio"] = df_full_train["remote_ratio"].replace({0:1, 50:2, 100:3})
categorical_columns = df_full_train.select_dtypes(include="object").columns.values
numerical_columns = df_full_train.select_dtypes(include="int64").columns.values
corr_matrix_plot = plt.figure(figsize=(10,4),dpi=200)
plt.title("Correlation Matrix between features")
corr_matrix = df_full_train[numerical_columns].corr()
heatmap = sns.heatmap(corr_matrix, annot=True)
heatmap.set(ylim=(0,len(corr_matrix)))
heatmap.set(xlim=(0,len(corr_matrix)))
with exp_data:
     st.title("Exploratory Data Analysis")
     st.subheader("Train data set")
     st.write(df_full_train.head())
     #bar plot job title distribution
     st.subheader("Barplot Job title distribution")     
     st.pyplot(job_title_bar_plot)
     #box plot salary vs job_title
     st.subheader("Salary vs Job Title")
     st.pyplot(job_title_box_plot)
     #box plot salary vs employment_type
     st.subheader("Salary vs Experience Level")
     st.pyplot(experience_box_plot)
     #box plot salary vs employment_type
     st.subheader("Barplot Experience Level distribution")
     st.pyplot(experience_box_plot)
     #Correlation matrix
     st.subheader("Correlation Matrix between features")
     st.pyplot(corr_matrix_plot)

###Section 3: Present Final Hyperparameters and RMSE in the test set
with final_model:
  st.title("Model Selection and Hyperparameters")
  st.write("""After trying different models, the best performance (smaller RMSE) was reached by the **Gradient Boosting** model.
           After doing a greed search, best hyperparameters were:
           """)
  st.markdown(""" 
  - *learning_rate*: 0.05
  - *max_depth*: 7  
  - *n_estimators*: 210
  - *max_features*: sqrt
   """)
  st.write("""Final model was train with the full train set and **RMSE** obteined with the test set was **44,313.81 USD**, which is not a good result. 
           The obtained RMSE represents (in average) more than half of the salary for the mean of the salaryes.
           It's hard to explain the reasons of this poor result, but if we see the boxplot netween salary and job_title and the one between 
           salary and experience level we shouldn't be much surprised. We have a lot of variance in this two features, and for the model is hard to find 
           a patern in the features. 
           """)

###Section 4: Choose you inputs and get prediction
with user_int:
  st.sidebar.subheader("Fill with your job information:")
  def add_input_ui():
    params = dict()
    params["work_year"] = st.sidebar.selectbox('Select work year', options=["2020", "2021", "2022", "2023"])
    params["experience_level"] = st.sidebar.selectbox('Select your experience level', options=['SE', 'EN', 'MI', 'EX'])
    params["employment_type"] = st.sidebar.selectbox('Select your employment type', options=['FT', 'CT', 'PT', 'FL'])
    params["job_title"] = st.sidebar.selectbox('Select your job title', options=['Data Analyst', 'Data Engineer', 'Machine Learning Engineer', 'Artificial Intelligence, Engineer', 'Data Scientist', 'Data Architect', 'Business Intelligence Engineer', 'Data and Analytics Manager'])
    params["remote_ratio"] = st.sidebar.selectbox('Select remote ratio', options=["0", "50", "100"])
    params["company_size"] = st.sidebar.selectbox('Select company size', options=["S", "M", "L"])
    params["employee_residence"] = st.sidebar.selectbox('Select your country of residence', options=['US', 'CO', 'UA', 'IT', 'SI', 'IN', 'GB', 'RO', 'ES', 'CA', 'GR', 'PT', 'FR', 'NL', 'LV', 'MU', 'DE', 'PL', 'AM', 'HR', 'TH', 'KR', 'EE', 'TR', 'PH', 'BR', 'QA', 'RU', 'KE', 'TN', 'GH', 'AU', 'BE', 'CH', 'AD', 'EC', 'PE', 'MX', 'MD', 'NG', 'SA', 'NO', 'AR', 'EG', 'UZ', 'GE', 'JP', 'ZA', 'HK', 'CF', 'FI', 'IE', 'IL', 'AT', 'SG', 'SE', 'KW', 'CY', 'BA', 'PK', 'LT', 'IR', 'AS', 'HU', 'CN', 'CR', 'CL', 'PR', 'DK', 'BO', 'DO', 'ID', 'AE', 'MY', 'HN', 'CZ', 'DZ', 'VN', 'IQ', 'BG', 'JE', 'RS', 'NZ', 'LU', 'MT'])
    params["company_location"] = st.sidebar.selectbox('Select company location', options=['US', 'CO', 'UA', 'SI', 'IN', 'GB', 'RO', 'ES', 'CA', 'GR', 'PT', 'FR', 'NL', 'LV', 'MU', 'DE', 'PL', 'RU', 'IT', 'KR', 'EE', 'CZ', 'CH', 'BR', 'QA', 'KE', 'DK', 'GH', 'SE', 'PH', 'AU', 'TR', 'AD', 'EC', 'MX', 'IL', 'NG', 'SA', 'NO' ,'AR', 'JP', 'ZA' ,'HK', 'CF', 'FI', 'IE', 'SG', 'TH', 'HR', 'AM', 'BA', 'PK', 'LT', 'IR', 'BS', 'HU', 'AT', 'PR', 'AS', 'BE', 'ID', 'EG', 'AE', 'MY', 'HN', 'DZ', 'IQ', 'CN', 'NZ', 'CL', 'MD', 'LU', 'MT'])
    return params
  params = add_input_ui()
  cat_cols = ["experience_level","employment_type","job_title","company_size","employee_residence", "company_location"]
  for c in cat_cols:
    print(params[c])
    params[c] = params[c].lower().replace(" ","_")
  params["work_year"] = int(params["work_year"].replace("2020", "1").replace("2021", "2").replace("2022", "3").replace("2023", "4"))
  params["experience_level"] = int(params["experience_level"].replace('se',"1").replace('en',"2").replace('mi',"3").replace('ex',"4"))
  params["company_size"] = int(params["company_size"].replace('s',"1").replace('m',"2").replace('l',"3"))
  params["remote_ratio"] = int(params["remote_ratio"].replace("0","1").replace("50","2").replace("100","3"))

  dv_input_file = "dv.bin"
  model_input_file = "final_model.bin"
  with open(dv_input_file, "rb") as dv_input_file:
    dv=pickle.load(dv_input_file)
  with open(model_input_file, "rb") as model_input_file:
    model=pickle.load(model_input_file)
  X = dv.transform(params)
  y_pred = np.round(model.predict(X),2)
  st.sidebar.subheader("Yearly salary prediction (in USD)")  
  st.sidebar.write(y_pred)