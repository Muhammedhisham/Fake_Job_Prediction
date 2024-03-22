

from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the model and vectorizer
with open('FakeJob.pkl', 'rb') as f:
    model = pickle.load(f)

with open('count_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        title = request.form['title']
        location = request.form['location']
        department = request.form['department']
        company_profile = request.form['company_profile']
        description = request.form['description']
        requirements = request.form['requirements']
        employment_type = request.form['employment_type']
        required_experience = request.form['required_experience']
        required_education = request.form['required_education']
        benefits = request.form['benefits']
        industry = request.form['industry']
        function = request.form['function']
        
        # Preprocess input data
        text_data = [title, location, department, company_profile, description, requirements,
                     employment_type, required_experience, required_education, benefits, industry, function]
        vectorized_data = vectorizer.transform(text_data)
        
        # Make prediction
        prediction = model.predict(vectorized_data)
        print('prediction output',prediction)
        
        prediction = int(prediction[0])
        
        # Determine prediction result
        prediction_result = "It is a FAKE Job" if prediction == 1 else "It is a GENUINE Job"
        
        return render_template('index.html', prediction=prediction_result)

if __name__ == '__main__':
    app.run(debug=True)
