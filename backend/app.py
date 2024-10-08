from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import pandas as pd
import os
import logging

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Use an absolute path for the upload folder
UPLOAD_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), 'uploads'))
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def read_file(file_path):
    _, file_extension = os.path.splitext(file_path)
    try:
        if file_extension.lower() == '.csv':
            df = pd.read_csv(file_path)
        elif file_extension.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            return None, "Unsupported file format"
        
        if df.empty:
            return None, "The file is empty"
        return df, None
    except pd.errors.EmptyDataError:
        return None, "The file is empty or has no parseable data"
    except Exception as e:
        return None, f"An error occurred while reading the file: {str(e)}"

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        app.logger.error('No file part in the request')
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        app.logger.error('No selected file')
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        app.logger.info(f'Saving file to: {file_path}')
        file.save(file_path)
        
        app.logger.info(f'File saved. Checking content...')
        with open(file_path, 'rb') as f:
            content = f.read()
            app.logger.info(f'File size: {len(content)} bytes')
            if len(content) == 0:
                os.remove(file_path)
                return jsonify({'error': 'The uploaded file is empty'}), 400
        
        df, error = read_file(file_path)
        if error:
            os.remove(file_path)
            app.logger.error(f'Error reading file: {error}')
            return jsonify({'error': error}), 400
        
        app.logger.info(f'File processed successfully. Columns: {df.columns.tolist()}')
        return jsonify({'columns': df.columns.tolist()}), 200
    else:
        app.logger.error('Invalid file type')
        return jsonify({'error': 'Invalid file type. Please upload a CSV or Excel file.'}), 400

@app.route('/train', methods=['POST'])
def train_models():
    data = request.json
    filename = data['file']
    input_columns = data['input_columns']
    output_column = data['output_column']
    
    app.logger.info(f'Training models with file: {filename}, input columns: {input_columns}, output column: {output_column}')
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(file_path):
        app.logger.error(f'File not found: {file_path}')
        return jsonify({'error': 'File not found. Please upload the file again.'}), 400
    
    try:
        df = pd.read_csv(file_path)  # or pd.read_excel() depending on the file type
    except pd.errors.EmptyDataError:
        app.logger.error('The file is empty or has no parseable data.')
        return jsonify({'error': 'The file is empty or has no parseable data.'}), 400
    except Exception as e:
        app.logger.error(f'An error occurred while reading the file: {str(e)}')
        return jsonify({'error': f'An error occurred while reading the file: {str(e)}'}), 400
    
    if df.empty:
        app.logger.error('The file contains no data.')
        return jsonify({'error': 'The file contains no data.'}), 400
    
    if not all(col in df.columns for col in input_columns + [output_column]):
        app.logger.error('Selected columns are not present in the file.')
        return jsonify({'error': 'Selected columns are not present in the file.'}), 400

    # Splitting the data into train and test sets
    X = df[input_columns]
    y = df[output_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(),
    }
    
    results = {}

    # Train and evaluate each model
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[model_name] = {
            'mean_squared_error': mse,
            'r2_score': r2
        }
    
    app.logger.info('Models trained successfully')
    
    return jsonify({'message': 'Models trained successfully', 'results': results}), 200

if __name__ == '__main__':
    app.run(debug=True)