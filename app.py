from flask import Flask, render_template, request, redirect, url_for
import csv
from joblib import load
import pandas as pd
import os

app = Flask(__name__)

uploaded_data = []
processed_data = []
processed_data_verification = []

def clear_data():
    global uploaded_data, processed_data, processed_data_verification
    uploaded_data = []
    processed_data = []
    processed_data_verification = []


@app.route('/', methods=['GET'])
def index():
    clear_data()
    return render_template('index.html',uploaded_data=uploaded_data,
                          processed_data=processed_data,
                          processed_data_verification=processed_data_verification)


@app.route('/load_table', methods=['GET', 'POST'])
def load_table():
    global uploaded_data, processed_data, processed_data_verification
    
    if request.method == 'POST':
        csv_file = request.files['csvFile']
        if csv_file and csv_file.filename.endswith('.csv'):
            csv_data = csv_file.read().decode('utf-8').splitlines()
            csv_reader = csv.reader(csv_data)
            uploaded_data = list(csv_reader)
            processed_data = None
            processed_data_verification = None
        return redirect(url_for('load_table'))

    return render_template('load_table.html', uploaded_data=uploaded_data,
                          processed_data=processed_data,
                          processed_data_verification=processed_data_verification)

@app.route('/process_table', methods=['GET','POST'])
def process_table():
    global uploaded_data, processed_data, processed_data_verification
    if uploaded_data:
      df = pd.DataFrame(uploaded_data[1:], columns=uploaded_data[0])
      x_data = df[['diagonal', 'height_left', 'height_right', 'margin_low', 'margin_up', 'length']]

      # Chemin vers le modèle
      model_path = os.path.join(app.root_path, 'modeles', 'logit_model.joblib')

      if os.path.exists(model_path):
          model_reg_logist = load(model_path)
          df["Prediction"] = model_reg_logist.predict(x_data)
          df["Probability"] = model_reg_logist.predict_proba(x_data)[:, 1].round(3)
          verification = []
          for i in df['Probability']:
              if i >= 0.5:
                  verification.append('Vrai Billet')
              else:
                  verification.append('Faux Billet')
          df['Verification'] = verification

          processed_data = df.to_dict('records')
          processed_data_verification = zip(df["id"], df['Verification'])

          uploaded_data = None
      return redirect(url_for('process_table'))   
    return render_template('process_table.html', uploaded_data=uploaded_data,
                           processed_data=processed_data,
                           processed_data_verification=processed_data_verification)



if __name__ == '__main__':
    app.run(debug=True)

