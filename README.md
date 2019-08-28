# Employee-Absenteeism









PROJECT REPORT
ON
ABSENTEEISM
BY
A. DILIP SIMHA























Contents

1.	Chapter1: introduction


1.1	Problem Statement…………………………………………………………………………………….3


2.	Chapter2: Methodology


a.	Pre-Processing………………………………………………………………………………………5


b.	Missing Value Analysis………………………………………………………………………….5


c.	Outlier Analysis…………………………………………………………………………………….6


3.	Chapter3: Model Selection

a.	Decision Tree ………………………………………………………………………………………9

b.	Linear Regression ………………………………………………………………………………  10


4.	Chapter: Conclusion

a.	Model Evaluation………………………………………………………………………………… 12

b.	Model Selection……………………………………………………………………………………12


5.	Chapter5: Solution to Problem Statement …………………………………………………………13

6.	Appendix
a.	R Code………………………………………………………………………………………………….14
b.	Python Code…………………………………………………………………………………………22

 
Chapter 1: Introduction

1.1 Problem Statement for Project 1
XYZ is a courier company. As we appreciate that human capital plays an important role in collection,
transportation and delivery. The company is passing through genuine issue of Absenteeism. The
company has shared it dataset and requested to have an answer on the following areas:
1. What changes company should bring to reduce the number of absenteeism?
2. How much losses every month can we project in 2011 if same trend of absenteeism
continues?
1.2 Variables
There are 21 variables in our data in which 20 are independent variables and 1 (Absenteeism time in
hours) is dependent variable. Since the type of target variable is continuous, this is a regression
problem.
Variable Information:
1. Individual identification (ID)
2. Reason for absence (ICD).
- Absences attested by the International Code of Diseases (ICD) stratified into 21
categories (I to XXI) as follows:
I Certain infectious and parasitic diseases
II Neoplasms
III Diseases of the blood and blood-forming organs and certain disorders involving the immune
mechanism
IV Endocrine, nutritional and metabolic diseases
V Mental and behavioural disorders
VI Diseases of the nervous system
VII Diseases of the eye and adnexa
VIII Diseases of the ear and mastoid process
IX Diseases of the circulatory system
X Diseases of the respiratory system
XI Diseases of the digestive system
XII Diseases of the skin and subcutaneous tissue
XIII Diseases of the musculoskeletal system and connective tissue
XIV Diseases of the genitourinary system
XV Pregnancy, childbirth and the puerperium
XVI Certain conditions originating in the perinatal period
XVII Congenital malformations, deformations and chromosomal abnormalities
XVIII Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified
XIX Injury, poisoning and certain other consequences of external causes
XX External causes of morbidity and mortalityXXI Factors influencing health status and contact with health services.
And 7 categories without (CID) patient follow-up (22), medical consultation (23), blood donation
(24), laboratory examination (25), unjustified absence (26), physiotherapy (27), dental consultation
(28).
3. Month of absence
4. Day of the week (Monday (2), Tuesday (3), Wednesday (4), Thursday (5), Friday (6))
5. Seasons (summer (1), autumn (2), winter (3), spring (4))
6. Transportation expense
7. Distance from Residence to Work (KMs)
8. Service time
9. Age
10. Work load Average/day
11. Hit target
12. Disciplinary failure (yes=1; no=0)
13. Education (high school (1), graduate (2), postgraduate (3), master and doctor (4))
14. Son (number of children)
15. Social drinker (yes=1; no=0)
16. Social smoker (yes=1; no=0)
17. Pet (number of pet)
18. Weight
19. Height
20. Body mass index
21. Absenteeism time in hours (target)
Chapter2: Methodology
a.	Pre-Processing
Data pre-processing is a data mining technique that involves transforming raw data into an understandable format. Real-world data is often incomplete, inconsistent, and/or lacking in certain behaviours or trends, and is likely to contain many errors. Data pre-processing is a proven method of resolving such issues.

b.	Missing Value Analysis
Missing value analysis helps address several concerns caused by incomplete data. If cases with missing values are systematically different from cases without missing values, the results can be misleading. Also, missing data may reduce the precision of calculated statistics because there is less information than originally planned. Another concern is that the assumptions behind many statistical procedures are based on complete cases, and missing values can complicate the theory required.
If a variable has more than 30% of its values missing, then those values can be ignored, or the column itself is ignored. In our case, none of the columns have a high percentage of missing values. The maximum missing percentage is 4.33% i.e., Body Mass Index column. The missing values have been computed using KNN imputation.

 





c.	Outlier Analysis
In statistics, an outlier is a data point that differs significantly from other observations.
Outliers, being the most extreme observations, may include the sample maximum or sample minimum, or both, depending on whether they are extremely high or low. However, the sample maximum and minimum are not always outliers because they may not be unusually far from other observations.
Imputing outlier values: Missing values obtained from boxplots are first converted to have NA values. Then these missing values are imputed using KNN imputation method.
 


 
 


 


















3.Model Selection
1.	Decision Tree
Decision Tree algorithm belongs to the family of supervised learning algorithms. Decision trees are used for both classification and regression problems. A decision tree is a tree where each node represents a feature(attribute), each link(branch) represents a decision(rule) and each leaf represents an outcome (categorical or continues value).  The general motive of using Decision Tree is to create a training model which can use to predict class or value of target variables by learning decision rules inferred from prior data (training data).


 












Plot of actual values vs predicted values
Figure below depicts the plot of actual values vs predicted values.
 

2.	Linear Regression 
Multiple linear regression is the most common form of linear regression analysis. Multiple linear regression is used to explain the relationship between one continuous dependent variable and two or more independent variables. The independent variables can be continuous or categorical.
Here the dependent variable or target variable, “Absenteeism time in hours” is a continuous variable hence linear regression is to be used.

 
Plot of actual values vs predicted values in linear regression model.

MAE(in percentage)	RMSE(in percentage)
6.36	13.42















Chapter4: Conclusion

a. Model Evaluation
In the previous chapter we have seen the Root Mean Square Error (RMSE) and MAE value for linear regression model. Root Mean Square Error (RMSE) is the standard deviation of the residuals (prediction errors). Residuals are a measure of how far from the regression line data points are, RMSE is a measure of how spread out these residuals are. In other words, it tells you how concentrated the data is around the line of best fit. 
As the square root of a variance, RMSE can be interpreted as the standard deviation of the unexplained variance and has the useful property of being in the same units as the response variable. 
MAE measures the average magnitude of the errors in a set of predictions, without considering their direction. It’s the average over the test sample of the absolute differences between prediction and actual observation where all individual differences have equal weight.
Comparison
Similarities: 
Both MAE and RMSE express average model prediction error in units of the variable of interest. Both metrics can range from 0 to ∞ and are indifferent to the direction of errors. They are negatively-oriented scores, which means lower values are better.
Differences: 
Taking the square root of the average squared errors has some interesting implications for RMSE. Since the errors are squared before they are averaged, the RMSE gives a relatively high weight to large errors. This means the RMSE should be more useful when large errors are particularly undesirable. 
From an interpretation standpoint, MAE is clearly the winner. RMSE does not describe average error alone and has other implications that are more difficult to tease out and understand.

c.	Model Selection
Keeping in view the case study problem and analysing the dependent variable, a classification method and regression method was adopted viz. decision tree and linear regression.
However, since the criteria in question is a clear case of linear regression as the predictor variable is “absenteeism in hours” which is a continuous variable and cannot be judged as a “Yes” or “No” scenario.




Chapter5: Solutions to the problem statement

        a. What changes company should bring to reduce the number of absenteeism?
It is seen in the linear regression model that employees with disciplinary action and those who put in more service time contribute majorly for increase in absenteeism in the company.
Also the raw data does not have gender data which does not give clarity for the medical reasons listed. Secondly the data consists of redundancy. The Hit target does not specifically say where its hours per day or week or month. Data dictionary is a missing factor in the raw data. Plethora of missing data though imputation methods have been used.
On analysis it is found that there are 2 females taking the reason cited as 15 and 16 which are pregnancy, pre natal and post natal. 
Age is also a factor for increase in absenteeism.
So the management might think of providing some behavioural trainings to their employees particularly apprising them of team work, legal importance particularly if they conduct against the agreed disciplinary code of conduct, managing stress, health programmes etc.
Employees who are above age of 45 can be moved to other office related work domains by giving them reasonable training.

b.How much losses every month can we project in 2011 if same trend of absenteeism continues?
This is a time series evaluation.  Preliminary analysis reveals that the peak absenteeism is in the month of March. This could be due to heavy load of work due to financial year ending agenda or the temperate weather conditions prevalent.












Appendix:

Codes to used 

R code
rm(list=ls())
getwd()
library("readxl")
## load the data
data = read_excel("Absenteeism_at_work_Project.xls")
##Class of the data
class(data)
data=data.frame(data)
##dimension of the data
dim(data)
##column names of the data
colnames(data)
## structure of the data to know the data types
str(data)
## all are numeric data types
## need to convert certain variables into categorical as denoted by 0 and 1
## redunt data is visible on inital analysis
## missing vale analysis to be done
### Convert numeric variables to categorical variables
data$Disciplinary.failure=as.factor(data$Disciplinary.failure)
data$Son=as.factor(data$Son)
data$Day.of.the.week=as.factor(data$Day.of.the.week)
data$Month.of.absence=as.factor(data$Month.of.absence)
data$Social.drinker=as.factor(data$Social.drinker)
data$Social.smoker=as.factor(data$Social.smoker)
data$Pet=as.factor(data$Pet)
data$Education=as.factor(data$Education)
data$Seasons=as.factor(data$Seasons)
### Removing Redunt Data
data=unique(data)
###
str(data)
#######################################################Missing value analysis########
sum(is.na(data))

missing_value = data.frame(apply(data,2,function(x){sum(is.na(x))}))

names(missing_value)[1] = "Missing_data"

missing_value$Variables = row.names(missing_value)

row.names(missing_value) = NULL

#Reaarange the columns
missing_value = missing_value[, c(2,1)]

#place the variables according to their number of missing values.
missing_value = missing_value[order(-missing_value$Missing_data),]

#Calculate the missing value percentage
missing_value$percentage = (missing_value$Missing_data/nrow(data) )* 100

##################Store the missing value information in a csv file############
write.csv(missing_value,"Project_Missing_value.csv", row.names = F)

## missing value percentage is less than 30 % so there is no need to drop the missing observations
### try out mean imputation methods by mean, median and knn methods and select the best method

## taking back up of data in case of any loss of data

df=data
data=df

## creating a missing value intentionally to validate the imputation method######__________________

data$Distance.from.Residence.to.Work[6]

## Actual Value=51
## Mean Value=29.24
## Median Value=26
## KNN Imputation=51

## Mean Value Imputation

data$Distance.from.Residence.to.Work[6]=NA
data$Distance.from.Residence.to.Work[is.na(data$Distance.from.Residence.to.Work)]=mean(data$Distance.from.Residence.to.Work,na.rm = T)
data$Distance.from.Residence.to.Work[6]

## Median Imputation
data$Distance.from.Residence.to.Work[6]=NA
data$Distance.from.Residence.to.Work[is.na(data$Distance.from.Residence.to.Work)]=median(data$Distance.from.Residence.to.Work,na.rm=T)
data$Distance.from.Residence.to.Work[6]

## KNN Imputation#######________________________________________

data$Distance.from.Residence.to.Work[6]=NA

library(DMwR)
data=knnImputation(data,k=3)
data$Distance.from.Residence.to.Work[6]

## KNN imputation gives the exact value so we choose KNN Method###########_______________

apply(data,2, function(x){sum(is.na(x))}) ### to check if all missing values are removed

## Conclusion by above method No missing values found

### Outlier analysis###################

numeric_index = sapply(data, is.numeric)### selecting variables which are only numeric, notice varibles gets reduced.
numeric_index

numeric_data = data[,numeric_index]### storing numeric variables
numeric_data

numeric_data = as.data.frame(numeric_data)### changing the variable type

cnames = colnames(numeric_data)[-12]### extracting the column names of numeric variables.

class(cnames)

## Box Plot to detect outliers####____________________________________________________

library(ggplot2)

for (i in 1:length(cnames)) {
  assign(paste0("gn",i), ggplot(aes_string( y = (cnames[i]), x= "Absenteeism.time.in.hours") , data = subset(data)) + 
           stat_boxplot(geom = "errorbar" , width = 0.5) +
           geom_boxplot(outlier.color = "red", fill = "grey", outlier.shape = 20, outlier.size = 1, notch = FALSE)+
           theme(legend.position = "bottom")+
           labs(y = cnames[i], x= "Absenteeism.time.in.hours")+
           ggtitle(paste("Boxplot" , cnames[i])))
           print(i)
}
# ## Plotting plots together#####___________________________________
install.packages("gridExtra")
gridExtra::grid.arrange(gn1,gn5,gn2,ncol=3)
gridExtra::grid.arrange(gn6,gn7,ncol=2)
gridExtra::grid.arrange(gn8,gn9,ncol=2)
gridExtra:: grid.arrange(gn10,gn11, ncol=2)

#### ### ###loop to remove outliers from all variables########_____________________________
 for(i in cnames){
 print(i)
 val = df[,i][df[,i] %in% boxplot.stats(df[,i])$out]
 print(length(val))
 df = df[which(!df[,i] %in% val),]
 }

## Replace all outliers with NA#######
for(i in numeric_data){
  val = df[,i][df[,i] %in% boxplot.stats(df[,i])$out]
  print(paste(i,length(val)))
  df[,i][df[,i] %in% val] = NA
}

#######Check number of missing values########

sapply(df,function(x){sum(is.na(x))})

#######################Compute the NA values using KNN imputation###########

df = knnImputation(df,k = 3)

###################Check if any missing values###########################

sum(is.na(df))


##### Correlation Plot####
install.packages("corrgram")
library(corrgram)

corrgram(data[,numeric_index], order = F,
         upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")#### to select important variables

### ################################Build Desiscion Tree#####################################################

set.seed(100)
train_index = sample(1:nrow(df), 0.8*nrow(df))### Simple random sampling method used as target variable is continous        
train = df[train_index,]
test = df[-train_index,]

##############################Build decsion tree using rpart#####################

library(rpart)
library(rpart.plot)
library(MASS)


dt_model = rpart(Absenteeism.time.in.hours~.,data = train, method = "anova")


##############################Plotting the decision tree########################################3
rpart.plot(dt_model)

##########################Perdict for test cases############################################
dt_predictions = predict(dt_model, test[,-115])

########################Create data frame for actual and predicted values####################
df_pred = data.frame("actual"= test[,-115], "dt_pred"=dt_predictions)
head(df_pred)

########################Calcuate MAE, RMSE, R-sqaured for testing data#######
print(postResample(pred = dt_predictions, obs = test[,-115]))

###############################Plot a graph for actual vs predicted values#####
plot(test$Absenteeism.time.in.hours,type="l",lty=2,col="red")

lines(dt_predictions,col="blue")

#######################Evaluate the performance of classification model

ConfMatrix_C50 = table(test$Absenteeism.time.in.hours, df_pred)
confusionMatrix(ConfMatrix_C50)

######################################## Linear Regression######################################

######Train the model using training data
l_model = lm(Absenteeism.time.in.hours ~ ., data = train)

#######Get the summary of the model

summary(l_model)

#Predict
l_predictions = predict(l_model,test)


#Plot a graph for actual vs predicted values
plot(test$Absenteeism.time.in.hours,type="l",lty=2,col="green")
lines(l_predictions,col="blue")### Test data predicting better than actual data

#####Error Matrix#####

#calculate MAE
library(DMwR)

regr.eval(test[,21],dt_predictions,stats=c('mae',"rmse"))

## MAE Value= 6.36% of error
### RMSE = 13.42% of error


Python code
In [ ]:
#Import libraries
import os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from fancyimpute import KNN
from sklearn.metrics import mean_squared_error

#import libraries for plots
import matplotlib.pyplot as plt
import seaborn as sns
% matplotlib inline
In [ ]:
# Setting working directory
os.chdir("C:/Users/Dilip/Desktop/Edwisor/Project")

# Loading data
emp_absent = pd.read_excel("Absenteeism_at_work.xls")
Exploratory Data Analysis
In [ ]:
emp_absent.shape
In [ ]:
# First 5 rows of data
emp_absent.head()
In [ ]:
# Data Types of all the variables
emp_absent.dtypes
In [ ]:
# Number of Unique values present in each variable
emp_absent.nunique()
In [ ]:
#Transform data types
emp_absent['ID'] = emp_absent['ID'].astype('category')

emp_absent['Reason for absence'] = emp_absent['Reason for absence'].replace(0,20)
emp_absent['Reason for absence'] = emp_absent['Reason for absence'].astype('category')

emp_absent['Month of absence'] = emp_absent['Month of absence'].replace(0,np.nan)
emp_absent['Month of absence'] = emp_absent['Month of absence'].astype('category')

emp_absent['Day of the week'] = emp_absent['Day of the week'].astype('category')
emp_absent['Seasons'] = emp_absent['Seasons'].astype('category')
emp_absent['Disciplinary failure'] = emp_absent['Disciplinary failure'].astype('category')
emp_absent['Education'] = emp_absent['Education'].astype('category')
emp_absent['Son'] = emp_absent['Son'].astype('category')
emp_absent['Social drinker'] = emp_absent['Social drinker'].astype('category')
emp_absent['Social smoker'] = emp_absent['Social smoker'].astype('category')
emp_absent['Pet'] = emp_absent['Pet'].astype('category')
In [ ]:
#Make a copy of dataframe
df = emp_absent.copy()
In [ ]:
# From the EDA and problem statement file categorising the variables in two category " Continuous" and "Categorical"
continuous_vars = ['Distance from Residence to Work', 'Service time', 'Age', 'Work load Average/day ', 'Transportation expense',
       'Hit target', 'Weight', 'Height', 'Body mass index', 'Absenteeism time in hours']

categorical_vars = ['ID','Reason for absence','Month of absence','Day of the week',
                     'Seasons','Disciplinary failure', 'Education', 'Social drinker',
                     'Social smoker', 'Pet', 'Son']
Missing Value Analysis
In [ ]:
#Creating dataframe with number of missing values
missing_val = pd.DataFrame(df.isnull().sum())

#Reset the index to get row names as columns
missing_val = missing_val.reset_index()

#Rename the columns
missing_val = missing_val.rename(columns = {'index': 'Variables', 0: 'Missing_perc'})
missing_val

#Calculate percentage
missing_val['Missing_perc'] = (missing_val['Missing_perc']/len(df))*100

#Sort the rows according to decreasing missing percentage
missing_val = missing_val.sort_values('Missing_perc', ascending = False).reset_index(drop = True)

#Save output to csv file
missing_val.to_csv("Missing_perc.csv", index = False)

Impute missing values
In [ ]:
## Actual Value=51
## Mean Value=29.24
## Median Value=26
## KNN Imputation=51
print(df[' 
Distance.from.Residence.to.Work'].iloc[6])

#Set the value of first row in Body mass index as NAN
#create missing value
df[' 
Distance.from.Residence.to.Work'].iloc[6]
] = np.nan
In [ ]:
#Impute with mean
#df['' 
Distance.from.Residence.to.Work']
= df[' Distance.from.Residence.to.Work'].fillna(df[' Distance.from.Residence.to.Work'].mean())

#Impute with median
#df[‘Distance.from.Residence.to.Work'] = df['Distance.from.Residence.to.Work'].fillna(df[Distance.from.Residence.to.Work''].median())

#Apply KNN imputation algorithm
df = pd.DataFrame(KNN(k = 3).fit_transform(df), columns = df.columns)
df[' 
Distance.from.Residence.to.Work'].iloc[6])
].iloc[1]
In [ ]:
#Round the values of categorical values
for i in categorical_vars:
    df.loc[:,i] = df.loc[:,i].round()    
    df.loc[:,i] = df.loc[:,i].astype('category')
In [ ]:
#Check if any missing values
df.isnull().sum()




Outlier Analysis
In [ ]:
#Check for outliers using boxplots
for i in continuous_vars:
    # Getting 75 and 25 percentile of variable "i"
    q75, q25 = np.percentile(df[i], [75,25])
    
    # Calculating Interquartile range
    iqr = q75 - q25
    
    # Calculating upper extream and lower extream
    minimum = q25 - (iqr*1.5)
    maximum = q75 + (iqr*1.5)
    
    # Replacing all the outliers value to NA
    df.loc[df[i]< minimum,i] = np.nan
    df.loc[df[i]> maximum,i] = np.nan


# Impute missing values with KNN
df = pd.DataFrame(KNN(k = 3).fit_transform(df), columns = df.columns)
# Checking if there is any missing value
df.isnull().sum()
In [ ]:
#Check for outliers in data using boxplot
sns.boxplot(data=df[['Absenteeism time in hours','Body mass index','Height','Weight']])
fig=plt.gcf()
fig.set_size_inches(8,8)
In [ ]:


#Check for outliers in data using boxplot
sns.boxplot(data=df[['Hit target','Service time','Age','Transportation expense']])
fig=plt.gcf()
fig.set_size_inches(8,8)

#Splitting data into train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( df.iloc[:, df.columns != 'Absenteeism time in hours'], df.iloc[:, 8], test_size = 0.20, random_state = 1)
Decision Tree
RMSE: 4.056

In [ ]:
# Importing libraries for Decision Tree 
from sklearn.tree import DecisionTreeRegressor

#Build decsion tree using DecisionTreeRegressor
dt_model = DecisionTreeRegressor(random_state = 1).fit(X_train,y_train)

#Perdict for test cases
dt_predictions = dt_model.predict(X_test)

#Create data frame for actual and predicted values
df_dt = pd.DataFrame({'actual': y_test, 'pred': dt_predictions})
print(df_dt.head())

#Define function to calculate RMSE
def RMSE(y_actual,y_predicted):
    rmse = np.sqrt(mean_squared_error(y_actual,y_predicted))


#Calculate RMSE and R-squared value
print("Root Mean Squared Error: "+str(RMSE(y_test, dt_predictions)))
Linear Regression
RMSE: 40145e+8
R-squared: -1.4181e+24
In [ ]:
# #########Importing libraries for Linear Regression
from sklearn.linear_model import LinearRegression

#Train the model
lr_model = LinearRegression().fit(X_train , y_train)

#Perdict for test cases
lr_predictions = lr_model.predict(X_test)

#Create data frame for actual and predicted values
df_lr = pd.DataFrame({'actual': y_test, 'pred': lr_predictions})
print(df_lr.head())

#Calculate RMSE and R-squared value
print("Root Mean Squared Error: "+str(RMSE(y_test, lr_predictions)))



