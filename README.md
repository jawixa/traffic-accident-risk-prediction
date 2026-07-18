# Traffic Accident Risk Prediction using Artificial Neural Networks

A machine learning project that predicts traffic accident risk using an Artificial Neural Network (MLPClassifier) trained on a carefully designed synthetic dataset.

The project combines statistical analysis, machine learning, and a Flask-based web application to provide an interactive traffic accident risk prediction system.

---

## Features

- Traffic accident risk prediction
- Artificial Neural Network (MLPClassifier)
- Interactive Flask web application
- Statistical data analysis
- Data visualization
- Synthetic dataset generation
- PostgreSQL database integration
- User-friendly interface

---

## Technologies

### Programming Language
- Python

### Machine Learning
- Scikit-learn
- MLPClassifier
- Adam Optimizer
- ReLU Activation Function

### Data Analysis
- Pandas
- NumPy
- Matplotlib

### Web Development
- Flask
- HTML
- CSS

### Database
- PostgreSQL

---

## Dataset

The model was trained using a carefully designed synthetic dataset.

### Dataset Characteristics

- 2,000 synthetic samples
- Balanced dataset (50% accident / 50% no accident)
- 14 input variables
- Dataset consistency verified using two different AI approaches

### Input Features

The prediction model uses the following parameters:

- Time Interval
- Temperature (°C)
- Visibility (m)
- Traffic Density
- Speed Limit
- Average Speed
- Intersection Density
- Rush Hour
- Day Type
- Rain Condition
- Road Type
- Road Surface
- Lighting Condition

---

## Machine Learning Model

- **Algorithm:** MLPClassifier
- **Activation Function:** ReLU
- **Optimizer:** Adam
- **Library:** Scikit-learn
- **Model Accuracy:** **94%**

The trained model is exported using Joblib and integrated into the Flask application for real-time predictions.

---

## Project Structure

```text
traffic-accident-risk-prediction/
│
├── dataset/
├── models/
├── src/
│   ├── mlp_model.py
│   ├── statistical_analysis.py
│   └── risk_prediction.py
│
├── traffic_app/
│   ├── app.py
│   ├── database.py
│   ├── static/
│   └── templates/
│
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/jawixa/traffic-accident-risk-prediction.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python traffic_app/app.py
```

Open your browser and visit:

```
http://127.0.0.1:5000
```

---

## Future Improvements

- Train using real-world traffic datasets
- Compare multiple machine learning algorithms
- Add REST API support
- Docker deployment
- Cloud deployment
- Improve user interface
- Feature importance analysis

---

## Developer

**Atay Mert Çalıcılar**

Statistics and Computer Science Graduate

- GitHub: https://github.com/jawixa
- LinkedIn: *(Atay Mert ÇALICILAR)*

---

## License

This project is licensed under the MIT License.
