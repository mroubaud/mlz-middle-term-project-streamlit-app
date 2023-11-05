import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import pickle

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.feature_extraction import DictVectorizer

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_squared_error

#load data
df = pd.read_csv("salaries.csv")
df=df.drop(["salary","salary_currency"], axis=1)

# Transform categorical jeraquical columns to ordinal
df["work_year"] = df["work_year"].replace({2020:1, 2021:2, 2022:3, 2023:4})
df["experience_level"] = df["experience_level"].replace({'se':1, 'en':2, 'mi':3, 'ex':4})
df["company_size"] = df["company_size"].replace({'s':1, 'm':2, 'l':3})
df["remote_ratio"] = df["remote_ratio"].replace({0:1, 50:2, 100:3})

#separate columns in numerical and categorical again
categorical_columns = df.select_dtypes(include="object").columns.values
numerical_columns = df.select_dtypes(include="int64").columns.values
for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(" ","_")

#function for grouping job_titles
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

#apply job_title function
df["job_title"] = df["job_title"].apply(classify_job_title)
df = df[df["job_title"] != 'other']

#delete salaries over 300.000 USD/year, we consider this outliers
df=df[(df["salary_in_usd"]<300000)]

#separate columns in numerical and categorical again
categorical_columns = df.select_dtypes(include="object").columns.values
numerical_columns = df.select_dtypes(include="int64").columns.values

#Split data in train, val and test sets
X=df.drop("salary_in_usd",axis=1)
y=df["salary_in_usd"]
X_full_train, X_test, y_full_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

#dict_vec
dv = DictVectorizer(sparse=False)
dicts_full_train = X_full_train.to_dict(orient='records')
X_full_train = dv.fit_transform(dicts_full_train)
dicts_test = X_test.to_dict(orient='records')
X_test = dv.transform(dicts_test)

#final model 
grad_boost_final_model = GradientBoostingRegressor(learning_rate=0.07, max_depth=7, max_features='sqrt', n_estimators=210)
grad_boost_final_model.fit(X_full_train, y_full_train)

#final model evaluation
y_pred = grad_boost_final_model.predict(X_test)
final_rmse = np.round(np.sqrt(mean_squared_error(y_test,y_pred)),2)
print("RMSE test set:", final_rmse)

#Saving the model
output_file = "final_model.bin"
with open(output_file, 'wb') as f_out: 
    pickle.dump(grad_boost_final_model, f_out)

#Saving the dic vec
output_file = "dv.bin"
with open(output_file, 'wb') as f_out: 
    pickle.dump(dv, f_out)