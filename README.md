# Diabetic Readmission

The project focuses on reducing the 30-day readmission rate for diabetic patients. Our objective is to leverage data analytics, predictive modeling, and visualization techniques to identify key factors contributing to readmission and enable the hospital to take proactive corrective actions.

## Data Analytics - Predictive Modeling
The project encompasses several key tasks. Firstly, we will analyze a provided dataset to determine the leading variables associated with readmission. By applying advanced analytical techniques and data mining methodologies, we will identify significant factors contributing to readmission, allowing the hospital to prioritize interventions and develop tailored strategies to reduce the readmission rate and enhance patient outcomes.

### Prerequisites: 
#### Softwares: 
- [Python 3.10.9](https://www.python.org/downloads/) or later for Windows, MacOS & Linux
- [Pip 21.3](https://pip.pypa.io/en/stable/) or later
#### Packages: 
```python
    # Default packages
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt  # Graphs & plots
    import seaborn as sns   # Data visualizations
    from lightgbm import LGBMClassifier # sklearn is for machine learning and statistical modeling including classification, regression, clustering and dimensionality reduction 

    from sklearn.model_selection import train_test_split, cross_validate   #break up dataset into train and test sets
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler, MinMaxScaler


    # Packages relating to missing data
    import missingno as msno
    import re    # This library is used to perform regex pattern matching

    # Required functions from sklearn
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import LinearSVC
    from sklearn.ensemble import GradientBoostingClassifier
    from catboost import CatBoostClassifier
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, classification_report, make_scorer
    from sklearn.linear_model import SGDClassifier
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split

    from sklearn.model_selection import KFold,cross_val_score, RepeatedStratifiedKFold,StratifiedKFold
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import OneHotEncoder,StandardScaler,PowerTransformer
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import KNNImputer,SimpleImputer
    from sklearn.compose import make_column_transformer
    from imblearn.pipeline import make_pipeline
    from sklearn.svm import SVC
    from sklearn.impute import SimpleImputer
    from sklearn.dummy import DummyClassifier
    from imblearn.over_sampling import SMOTE
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score,\
                                precision_score, recall_score, roc_auc_score,\
                                plot_confusion_matrix, classification_report, plot_roc_curve, f1_score

    import plotly 
    import plotly.express as px
    import plotly.graph_objs as go
    import plotly.offline as py
    from plotly.offline import iplot
    from plotly.subplots import make_subplots
    import plotly.figure_factory as ff
```

## Infrastructure
Secondly, we will design the infrastructure components required to build, test, and deploy an algorithm to predict the likelihood of 30-day readmissions in the diabetic population. By utilizing the technologies and methodologies learned, we will create a robust workflow that integrates the predictive algorithm seamlessly into the existing infrastructure. This will enable the hospital to identify high-risk patients and initiate timely interventions to prevent readmissions.

## Visualization
Lastly, we will visualize the data analytics from the diabetes readmission predictive model. Through comprehensive visualizations in Tableau, we will highlight the crucial factors influencing readmission, providing actionable insights for the hospital to implement targeted interventions.

Throughout the project we utilized our expertise in data analytics, predictive modeling, and visualization will drive effective decision-making and facilitate a reduction in the 30-day readmission rate for the diabetic population.