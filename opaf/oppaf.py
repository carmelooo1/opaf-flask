from flask import Flask, render_template, request
import pandas as pd
import csv
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

data = pd.read_csv('data/datasets1.csv')

X = data[['Math', 'Science', 'English', 'AP', 'TLE', 'MAPEH', 'Filipino', 'VALUES', 'Gender']]
y = data['Primary']
z = data['Secondary']

model1 = DecisionTreeClassifier()
model1.fit(X, y)

model2 = DecisionTreeClassifier()
model2.fit(X, z)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit_grades', methods=['GET', 'POST'])
def submit_grades():
    if request.method == 'POST':
        math_grade = (request.form['math'])
        science_grade = (request.form['science'])
        english_grade = (request.form['english'])
        ap_grade = (request.form['ap'])
        values_grade = (request.form['values'])
        tle_grade = (request.form['tle'])
        mapeh_grade = (request.form['mapeh'])
        filipino_grade =(request.form['filipino'])
        gender_id = (request.form['gender'])
        
        predicted_strand = model1.predict([[math_grade, science_grade, english_grade, ap_grade, 
                                        values_grade, tle_grade, mapeh_grade, filipino_grade, gender_id]])
        predicted_strand1 = model2.predict([[math_grade, science_grade, english_grade, ap_grade, 
                                        values_grade, tle_grade, mapeh_grade, filipino_grade, gender_id]])
        data_to_append = [math_grade, science_grade, english_grade, ap_grade, values_grade,
                          tle_grade, mapeh_grade, filipino_grade, gender_id, predicted_strand[0], predicted_strand1[0]]
        
        grades = [math_grade, science_grade, english_grade, ap_grade, 
                                        values_grade, tle_grade, mapeh_grade, filipino_grade]
           
                
        grades = [float(grade) for grade in grades]        
        average = sum(grades)/len(grades)
        sumave = str(int(average)) + '%'

            
        
        with open('data/datasets1.csv', 'a') as file:  
            writer = csv.writer(file)
            writer.writerow(data_to_append)
        
        return render_template('result.html',sumave = sumave, predicted_strand=predicted_strand[0], predicted_strand1=predicted_strand1[0])
    else:
        return render_template('submit.html')

if __name__ == '__main__':
    app.run(debug=True)
