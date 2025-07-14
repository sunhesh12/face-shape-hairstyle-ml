

For below part in the app.py file add your saved model

#Load the trained face shape classifier
model = joblib.load('D:/5th Sem/Machine Learning/ML Project/HairstylePredicAccordingToFaceshape/notebooks/SVM_Classification/SVM_Model2.pkl') 

After modifying the above part go to frontend folder in terminal and give the command "streamlit run app.py"

Then the app will run on a browser.

And upload a face image and check.