from flask import Flask, request, render_template_string
import pandas as pd
import joblib
import base64

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.joblib')

# Define the features
features = ['abnormal_short_term_variability', 'mean_value_of_short_term_variability',
            'percentage_of_time_with_abnormal_long_term_variability', 'histogram_mean', 'histogram_mode']

# HTML template as a string
html_template = """
<html>
<head>
   <style>
      /* Bootstrap CSS (Embedded) */
      body {
         margin: 0;
         padding: 0;
         font-family: Arial, sans-serif;
         background-color: #f7f7f7;
      }

      .parallax {
         background-image: url('data:image/jpeg;base64,{{image_data}}');
         height: 80vh;
         background-attachment: fixed;
         background-position: center;
         background-repeat: no-repeat;
         background-size: cover;
      }
      
      .container {
         margin-top: 20px;
         background: #0056b3;
         padding: 20px;
         color: white;
         border-radius: 10px;
         box-shadow: 0 0 15px rgba(0, 0, 0, 0.2);
      }

      .form-group {
         margin-bottom: 15px;
      }
      
      .purple-border {
         border-color: #82ABD3;
         border-width: 2px;
      }

      .btn-primary {
         background-color: #28a745;
         border-color: #28a745;
         color: white;
         font-size: 18px;
         padding: 10px 20px;
         border-radius: 5px;
         transition: background-color 0.3s ease;
      }

      .btn-primary:hover {
         background-color: #218838;
         border-color: #1e7e34;
      }

      .form-control {
         color: black;
         font-size: 18px; 
         padding: 10px;
         border-radius: 5px;
      }

      .result-box {
         background-color: #ffffff;
         color: #212529;
         padding: 15px;
         margin-top: 20px;
         border-radius: 5px;
         box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
         font-size: 18px;
      }

      .header {
         background-color: #0056b3;
         color: white;
         font-size: 36px;
         padding: 20px;
         text-align: center;
         border-radius: 10px;
         margin-top: 10px;
      }

      .custom-bg {
         background-color: #f0f0f0; 
         text-align: center;
         margin-top: 15px;
      }

      label {
         color: white;
         font-size: 18px;
      }

      textarea {
         width: 100%;
         resize: none;
      }
   </style>
</head>
<body>
   <div class="parallax"></div>
   <div class="header">Fetal Health Prediction</div>
   <div class="container">
      <form action="/" method="POST">
         <div class="form-group purple-border">
            <label for="query1">Abnormal Short Term Variability:</label>
            <textarea class="form-control" rows="1" id="query1" name="query1">{{query1}}</textarea>
         </div>
         <div class="form-group purple-border">
            <label for="query2">Mean Value of Short Term Variability:</label>
            <textarea class="form-control" rows="1" id="query2" name="query2">{{query2}}</textarea>
         </div>
         <div class="form-group purple-border">
            <label for="query3">Percentage of Time with Abnormal Long Term Variability:</label>
            <textarea class="form-control" rows="1" id="query3" name="query3">{{query3}}</textarea>
         </div>
         <div class="form-group purple-border">
            <label for="query4">Histogram Mean:</label>
            <textarea class="form-control" rows="1" id="query4" name="query4">{{query4}}</textarea>
         </div>
         <div class="form-group purple-border">
            <label for="query5">Histogram Mode:</label>
            <textarea class="form-control" rows="1" id="query5" name="query5">{{query5}}</textarea>
         </div>
         <button type="submit" class="btn btn-primary">Submit</button>
      </form>

      {% if output1 %}
         <div class="result-box">
            <p><strong>Prediction:</strong> {{output1}}</p>
            <p><strong>Confidence:</strong> {{output2}}</p>
         </div>
      {% endif %}
   </div>
</body>
</html>
"""

@app.route("/", methods=['GET'])
def loadPage():
    image_path = 'fetuss.jpeg'  # Corrected image path
    with open(image_path, 'rb') as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')
    return render_template_string(html_template, query1="", query2="", query3="", query4="", query5="", output1="", output2="", image_data=image_data)

@app.route("/", methods=['POST'])
def fetalHealthPrediction():
    if request.method == 'POST':
        try:
            # Convert input values to floats
            inputQuery1 = float(request.form['query1'])
            inputQuery2 = float(request.form['query2'])
            inputQuery3 = float(request.form['query3'])
            inputQuery4 = float(request.form['query4'])
            inputQuery5 = float(request.form['query5'])

            data = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5]]
            new_df = pd.DataFrame(data, columns=features)

            single = model.predict(new_df)
            probability = model.predict_proba(new_df)

            if single == 1:
                output = "The fetus is Normal"
                output1 = f"Confidence: {probability[0][0]*100:.2f}%"
            elif single == 2:
                output = "The fetus is Suspect"
                output1 = f"Confidence: {probability[0][1]*100:.2f}%"
            elif single == 3:
                output = "The fetus is Pathological"
                output1 = f"Confidence: {probability[0][2]*100:.2f}%"
            else:
                output = "Prediction error"
                output1 = "N/A"

            image_path = 'fetuss.jpeg'  # Corrected image path
            with open(image_path, 'rb') as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            return render_template_string(html_template, output1=output, output2=output1,
                                          query1=request.form['query1'], query2=request.form['query2'],
                                          query3=request.form['query3'], query4=request.form['query4'],
                                          query5=request.form['query5'], image_data=image_data)
        
        except Exception as e:
            print(f"Error: {e}")
            return render_template_string(html_template, output1="Error occurred", output2=str(e),
                                          query1=request.form['query1'], query2=request.form['query2'],
                                          query3=request.form['query3'], query4=request.form['query4'],
                                          query5=request.form['query5'], image_data=image_data)

if __name__ == "__main__":
    app.run(port=8080, debug=True)
