Handwritten Number Recognition AI

AI-powered handwritten number recognition using CNN, TensorFlow, OpenCV and Flask.



📌 Project Overview

Handwritten Number Recognition AI is a web-based computer vision and deep learning application that recognizes handwritten single and multi-digit numbers drawn on an interactive canvas. The system uses a Convolutional Neural Network trained on the MNIST dataset, combined with OpenCV preprocessing and a Flask web application.

✨ Features

•	Interactive drawing canvas using mouse or touchscreen

•	Single-digit recognition

•	Double-digit recognition

•	Triple-digit and multi-digit recognition

•	Support for numbers up to 10 detected digits

•	CNN-based handwritten digit classification

•	Per-digit confidence scores

•	Per-digit probability distribution from 0 to 9

•	Overall prediction confidence

•	Automatic digit segmentation

•	MNIST-style image preprocessing

•	Aspect-ratio preservation

•	Center-of-mass correction

•	Multiple preprocessing variants for improved prediction

•	Prediction history

•	SQLite database integration

•	Prediction analytics and digit distribution

•	Responsive and professional web interface

•	REST-style Flask endpoints

🖥️ Application Preview

Add your dashboard screenshot to the GitHub repository using the path below:

screenshots/dashboard.png

After uploading the screenshot, the Markdown version in README.md can display it with:

<p align="center">\\n<img src="screenshots/dashboard.png" width="900">\\n</p>

🧠 Machine Learning Model

The recognition engine uses a Convolutional Neural Network trained on the MNIST handwritten digit dataset. The model classifies digits from 0 through 9.

Dataset

Property	Value

Training images	60,000

Testing images	10,000

Input size	28 × 28 pixels

Classes	10, digits 0 through 9

Model type	Convolutional Neural Network

Framework	TensorFlow / Keras

Reported MNIST test accuracy	99.14%

🏗️ Recognition Pipeline

User Drawing

&#x20;    │

&#x20;    ▼

Canvas Image

&#x20;    │

&#x20;    ▼

Grayscale Conversion

&#x20;    │

&#x20;    ▼

Noise Removal

&#x20;    │

&#x20;    ▼

Thresholding

&#x20;    │

&#x20;    ▼

Contour Detection

&#x20;    │

&#x20;    ▼

Digit Segmentation

&#x20;    │

&#x20;    ▼

Tight Cropping

&#x20;    │

&#x20;    ▼

Aspect-Ratio Preservation

&#x20;    │

&#x20;    ▼

28 × 28 MNIST Normalization

&#x20;    │

&#x20;    ▼

Center-of-Mass Correction

&#x20;    │

&#x20;    ▼

Multiple Preprocessing Variants

&#x20;    │

&#x20;    ▼

CNN Prediction

&#x20;    │

&#x20;    ▼

Per-Digit Probabilities

&#x20;    │

&#x20;    ▼

Final Number

🛠️ Technologies Used

•	Python

•	TensorFlow

•	Keras

•	OpenCV

•	NumPy

•	Pillow

•	Flask

•	SQLite

•	HTML5

•	CSS3

•	JavaScript

•	HTML Canvas

📁 Project Structure

Handwritten-Digit-Recognition/

│

├── model/

│   └── digit\_model.keras

│

├── screenshots/

│   └── dashboard.png

│

├── static/

│   ├── script.js

│   └── style.css

│

├── templates/

│   └── index.html

│

├── .gitignore

├── LICENSE

├── README.md

├── app.py

├── requirements.txt

└── train.py

⚙️ Installation

1\. Clone the repository

git clone https://github.com/Pratikfating123/handwritten-number-recognition-ai.git

2\. Enter the project directory

cd handwritten-number-recognition-ai

3\. Create a virtual environment

python -m venv venv

4\. Activate the virtual environment

.\\venv\\Scripts\\Activate.ps1

5\. Install dependencies

pip install -r requirements.txt

▶️ Run the Application

python app.py

Open the following address in your browser:

http://127.0.0.1:5000

🧪 Train the Model

To retrain the CNN model using the MNIST dataset, run:

python train.py

The trained model is saved to:

model/digit\_model.keras

📊 API Endpoints

Endpoint	Purpose

GET /	Web application

POST /predict	Upload a drawing and receive a prediction

GET /history	Retrieve prediction history

POST /history/clear	Clear prediction history

GET /analytics	Retrieve prediction analytics

GET /health	Check application and model status

GET /api/security	Application/model status endpoint

GET /api/interfaces	Application interface information

🎯 Machine Learning Concepts Demonstrated

•	Convolutional Neural Networks

•	Image classification

•	Deep learning

•	Computer vision

•	Image preprocessing

•	Data augmentation

•	Feature extraction

•	Probability prediction

•	Model evaluation

•	Digit segmentation

•	Center-of-mass normalization

•	REST API development

•	Database integration

🔮 Future Improvements

•	Custom handwriting dataset for improved real-world accuracy

•	Handwriting-style adaptation

•	Real-time prediction while drawing

•	Improved multi-digit segmentation

•	Prediction screenshot export

•	Model comparison and benchmarking

•	Docker support

•	Cloud deployment

•	Mobile-friendly deployment

•	AI-generated explanation of predictions

📄 Requirements

tensorflow==2.21.0

flask==3.1.3

numpy==2.5.2

pillow==12.3.0

opencv-python==5.0.0.93

🔐 Privacy and Local Data

Prediction history is stored locally in SQLite. The local database file is excluded from Git using .gitignore, so personal prediction history is not uploaded to the repository.

👨‍💻 Author

Pratik Fating

MCA Cybersecurity Student

GitHub: https://github.com/Pratikfating123

LinkedIn: https://www.linkedin.com/in/pratik-fating-153448273/

⭐ Support

If you find this project useful, consider giving the repository a star on GitHub.

📜 License

This project is released under the MIT License.



