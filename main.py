from flask import Flask, render_template, request, send_from_directory, redirect, url_for, session
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import pandas as pd
import mysql.connector
from werkzeug.security import generate_password_hash, check_password_hash
import matplotlib.pyplot as plt
import tensorflow as tf

app = Flask(__name__)
app.secret_key = "ssk_secret_key"

# ---------- Upload Folder ----------
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ---------- MySQL Initialization ----------
def initialize_database():
    conn = mysql.connector.connect(host="localhost", user="root", password="#Darling09")
    cursor = conn.cursor()
    cursor.execute("CREATE DATABASE IF NOT EXISTS user_db")
    cursor.close()
    conn.close()

    db = mysql.connector.connect(host="localhost", user="root", password="#Darling09", database="user_db")
    cursor = db.cursor()

    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
        id INT AUTO_INCREMENT PRIMARY KEY,
        username VARCHAR(100) UNIQUE NOT NULL,
        email VARCHAR(100) UNIQUE NOT NULL,
        password_hash VARCHAR(255) NOT NULL)''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS contacts (
        id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(100) NOT NULL,
        email VARCHAR(100) NOT NULL,
        message TEXT NOT NULL)''')

    db.commit()
    cursor.close()
    return db

db = initialize_database()

# ---------- Model Setup ----------
model = load_model('models/model.h5')
class_labels = ['glioms', 'meningioma', 'notumor', 'ptuitary']

# Ensure model is initialized
dummy_input = np.zeros((1, 128, 128, 3), dtype=np.float32)
_ = model.predict(dummy_input)

# ---------- Prediction ----------
def predict_tumor(image_path):
    img = load_img(image_path, target_size=(128, 128))
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    pred = model.predict(arr)
    label_index = np.argmax(pred)
    confidence = np.max(pred)
    label = class_labels[label_index]
    return "No Tumor" if label == 'notumor' else label.capitalize(), confidence

# ---------- Grad-CAM ----------
def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in model.")

def find_layer_by_name(model, name):
    for layer in model.layers:
        if layer.name == name:
            return layer
        elif hasattr(layer, 'layers'):
            nested = find_layer_by_name(layer, name)
            if nested: return nested
    return None

def generate_grad_cam(image_path):
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    _ = model.predict(img_array)

    last_conv_layer_name = get_last_conv_layer(model)
    last_conv_layer = find_layer_by_name(model, last_conv_layer_name)

    grad_model = tf.keras.models.Model([model.input], [last_conv_layer.output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8

    # Save Grad-CAM
    plt.figure(figsize=(5, 5))
    plt.imshow(load_img(image_path))
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.axis('off')
    gradcam_path = os.path.join('static', 'gradcam.jpg')
    plt.savefig(gradcam_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    return gradcam_path

# ---------- Authentication ----------
@app.route('/register', methods=['GET', 'POST'])
def register():
    error, success = None, None
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password_hash = generate_password_hash(request.form['password'])
        cursor = db.cursor()
        cursor.execute("SELECT * FROM users WHERE username=%s OR email=%s", (username, email))
        if cursor.fetchone():
            error = "Username or email already exists."
        else:
            cursor.execute("INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s)",
                           (username, email, password_hash))
            db.commit()
            success = "Registration successful!"
        cursor.close()
    return render_template('register.html', error=error, success=success)

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        cursor = db.cursor()
        cursor.execute("SELECT password_hash FROM users WHERE username=%s", (username,))
        user = cursor.fetchone()
        if user and check_password_hash(user[0], password):
            session['user'] = username
            return redirect(url_for('home'))
        else:
            error = "Invalid login credentials."
        cursor.close()
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

@app.route('/home', methods=['GET', 'POST'])
def home():
    success = None
    if request.method == 'POST':
        name, email, message = request.form['name'], request.form['email'], request.form['message']
        cursor = db.cursor()
        cursor.execute("INSERT INTO contacts (name, email, message) VALUES (%s, %s, %s)", (name, email, message))
        db.commit()
        cursor.close()
        success = "Message sent successfully!"
    return render_template('home.html', success=success)

# ---------- Index Route (Prediction) ----------
@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            label, confidence = predict_tumor(filepath)
            gradcam_path = generate_grad_cam(filepath)

            df = pd.DataFrame([{
                "filename": file.filename,
                "predicted_label": label,
                "confidence": f"{confidence*100:.2f}%"
            }])
            log_path = 'predictions.csv'
            if os.path.exists(log_path):
                df.to_csv(log_path, mode='a', header=False, index=False)
            else:
                df.to_csv(log_path, index=False)

            return render_template('index.html',
                                   result=label,
                                   confidence=f"{confidence*100:.2f}%",
                                   file_path=f"/uploads/{file.filename}",
                                   gradcam_path=gradcam_path)
    return render_template('index.html', result=None)

@app.route('/overview')
def overview():
    log_path = 'predictions.csv'
    tumor_labels = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]
    tumor_counts = {label: 0 for label in tumor_labels}
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        counts = df['predicted_label'].value_counts()
        for label in tumor_labels:
            tumor_counts[label] = int(counts.get(label, 0))
    return render_template('overview.html',
                           tumor_counts=tumor_counts,
                           tumor_labels=list(tumor_counts.keys()),
                           tumor_values=list(tumor_counts.values()))

@app.route('/graphs')
def graphs():
    log_path = 'predictions.csv'
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        tumor_labels = df['predicted_label'].value_counts().index.tolist()
        tumor_counts = df['predicted_label'].value_counts().tolist()
        prediction_rows = df.tail(20).to_dict(orient='records')
    else:
        tumor_labels, tumor_counts, prediction_rows = [], [], []
    return render_template('graphs.html',
                           tumor_labels=tumor_labels,
                           tumor_counts=tumor_counts,
                           prediction_rows=prediction_rows)

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

@app.route('/model_summary')
def model_summary():
    summary_lines = []
    model.summary(print_fn=lambda x: summary_lines.append(x))
    return "<pre>" + "\n".join(summary_lines) + "</pre>"

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/')
def root():
    return redirect(url_for('login'))

# ---------- Run App ----------
if __name__ == '__main__':
    app.run(debug=True)
