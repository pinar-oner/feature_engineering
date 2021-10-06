FEATURE ENGINEERING 

Business Problem: 
A data preprocessing and feature engineering script for a machine learning pipeline needs to be prepared. It is expected that the dataset will be ready for modelling when passed through this script.  

Story of the Dataset:  
The dataset is the dataset of the people who were in the Titanic shipwreck. 
It consists of 768 observations and 12 variables. 
The target variable is specified as "Survived"; 

0: indicates the person's inability to survive. 

1: refers to the survival of the person. 

ATTRIBUTES: 

PassengerId: ID of the passenger 

Survived: Survival status (0: not survived, 1: survived) 

Pclass: Ticket class (1: 1st class (upper), 2: 2nd class (middle), 3: 3rd class(lower)) 

Name: Name of the passenger 

Sex: Gender of the passenger (male, female) 

Age: Age in years 

Sibsp: Number of siblings/spouses aboard the Titanic    
Sibling = Brother, sister, stepbrother, stepsister     
Spouse = Husband, wife (mistresses and fiances were ignored) 

Parch: Number of parents/children aboard the Titanic    
Parent = Mother, father     
Child = Daughter, son, stepdaughter, stepson  
Some children travelled only with a nanny , therefore Parch = 0 for them. 

Ticket: Ticket number # Fare: Passenger fare 

Cabin: Cabin number 

Embarked: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

REFERENCE: 
Data Science and ML Boot Camp, 2021, Veri Bilimi Okulu (https://www.veribilimiokulu.com/)
