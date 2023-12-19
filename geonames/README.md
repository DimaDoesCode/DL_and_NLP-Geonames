# Diploma Project

## Project Description: Customer Churn Prediction for TeleDom Communication Operator

TeleDom, a telecommunications company, aims to combat customer churn. To achieve this, its staff will start offering promo codes and special conditions to those planning to terminate their communication services. To proactively identify such users, TeleDom needs a model that predicts whether a subscriber will terminate their contract. The operator's team has gathered personal data about some customers, along with information about their tariffs and services. Your task is to train a model on this data to predict customer churn.

**Description of Services:**

The operator provides two main types of services:

1. **Landline Telephone Service:** Multiple lines can be connected simultaneously.

2. **Internet:** Connection can be through a telephone line (DSL - Digital Subscriber Line) or fiber optic cable.

Additional services include:

- Internet Security: antivirus (DeviceProtection) and blocking of unsafe websites (OnlineSecurity).
- Dedicated technical support line (TechSupport).
- Cloud file storage for data backup (OnlineBackup).
- Streaming TV (StreamingTV) and movie catalog (StreamingMovies).

Customers can pay for services monthly or enter into contracts for 1-2 years. Payment can be made in various ways, and electronic receipts are available.

## Data Description:

The data consists of several files obtained from different sources:

1. `contract_new.csv` — Contract information.
2. `personal_new.csv` — Customer's personal data.
3. `internet_new.csv` — Information about internet services.
4. `phone_new.csv` — Information about telephone services.

**contract_new.csv File:**

- `customerID` — Subscriber identifier.
- `BeginDate` — Contract start date.
- `EndDate` — Contract end date.
- `Type` — Payment type: annually-biannually or monthly.
- `PaperlessBilling` — Electronic billing.
- `PaymentMethod` — Payment method.
- `MonthlyCharges` — Monthly expenses.
- `TotalCharges` — Total subscriber charges.

**personal_new.csv File:**

- `customerID` — User identifier.
- `gender` — Gender.
- `SeniorCitizen` — Whether the subscriber is a senior citizen.
- `Partner` — Whether the subscriber has a spouse.
- `Dependents` — Whether the subscriber has dependents.

**internet_new.csv File:**

- `customerID` — User identifier.
- `InternetService` — Connection type.
- `OnlineSecurity` — Blocking of unsafe sites.
- `OnlineBackup` — Cloud file storage for data backup.
- `DeviceProtection` — Antivirus.
- `TechSupport` — Dedicated technical support line.
- `StreamingTV` — Streaming TV.
- `StreamingMovies` — Movie catalog.

**phone_new.csv File:**

- `customerID` — User identifier.
- `MultipleLines` — Connection of phone to multiple lines simultaneously.

In all files, the `customerID` column contains the client code.

Contract information is current as of February 1, 2020.

**Quality Metrics:**

The primary metric for model quality assessment is ROC_AUC. The interpretation metric is accuracy.

---

<a href="https://github.com/DimaDoesCode/Yandex_Practicum-Diploma_Project/blob/master/diploma_project/Diploma_Project_final.ipynb">To view the Jupyter Notebook code of the research, click on this link.</a>

## Libraries used
<i>catboost, datetime, IPython.display, joblib, json, matplotlib.pyplot, np, os, pandas, phik, seaborn, shap, sklearn, warnings</i>