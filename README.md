# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
Developed by:srimathi.s
RegisterNumber:25011379
```
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Example features: Satisfaction, Evaluation, Projects, Hours, Tenure → Churn (1=Left, 0=Stayed)
data = {
    'Satisfaction_Level': [0.8, 0.4, 0.9, 0.3, 0.7, 0.2, 0.85, 0.5, 0.95, 0.45],
    'Last_Evaluation': [0.9, 0.6, 0.8, 0.5, 0.75, 0.4, 0.88, 0.7, 0.95, 0.65],
    'Number_of_Projects': [5, 3, 6, 2, 4, 2, 6, 3, 7, 3],
    'Average_Monthly_Hours': [220, 150, 250, 120, 200, 100, 240, 160, 260, 140],
    'Years_at_Company': [3, 2, 4, 1, 3, 1, 4, 2, 5, 2],
    'Churn': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)
print("Employee Data:\n", df, "\n")


X = df[['Satisfaction_Level', 'Last_Evaluation', 'Number_of_Projects',
        'Average_Monthly_Hours', 'Years_at_Company']]
y = df['Churn']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))


plt.figure(figsize=(12, 6))
plot_tree(model,
          feature_names=X.columns,
          class_names=['Stayed', 'Left'],
          filled=True,
          rounded=True,
          fontsize=10)
plt.title("Decision Tree for Employee Churn Prediction")
plt.show()


new_emp = [[0.4, 0.6, 3, 150, 2]]  # Example new employee data
prediction = model.predict(new_emp)
print("\nNew Employee Prediction:")
if prediction[0] == 1:
    print(" Employee is likely to LEAVE (Churn).")
else:
    print(" Employee is likely to STAY.")
```

## Output:
<img width="1019" height="829" alt="image" src="https://github.com/user-attachments/assets/11866865-81f9-4a35-b26e-1d2132bb97d6" />
<img width="1140" height="824" alt="image" src="https://github.com/user-attachments/assets/2f90479d-ed99-4344-8d8c-ec12eb4c6c6c" />

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
