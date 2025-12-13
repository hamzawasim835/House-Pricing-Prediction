# 🏠Housing Price Predicition Project


.. By [Şule Ahmet](https://github.com/suleahmet) and [Hamza Darwish](https://github.com/hamzawasim835)  
Welcome to our House pricing prediction project. Using a tuned XgBoost model, this project takes in a number of inputs (house size, room number, etc) from the user through an HTML and JS based frontend and routes the data through an API to the model and gives the user an output, denoted in TRY, through the API and frontend.  


# 📋 About the project  
**Model:** Tuned XgBoost regression model.  

**Objective:** Predict the price of a real estate in Türkiye depending on 15 different factors.   

**Evaluation Criteria:** Mean Average Error (MAE), Real Mean Squared Error (RMSE), and R^2.  

**Deployment Platform:** Render, for both the API (Deployed as a web service), and the Frontend (Deployed as a static website).  

# 💻 Tools Used   
| Category | Tool | Purpose |
|--------|-----------|---------|
| Programming Language | Python | Core language used for data processing, modeling, and API development, in addition to Jupyter notebooks |
| Data Analysis | Pandas | Data cleaning, transformation, and feature engineering |
| Numerical Computing | NumPy | Efficient numerical operations and array handling |
| Machine Learning | XGBoost | Regression model for house price prediction |
| Model Persistence | Joblib | Saving and loading trained machine learning models |
| API Framework | FastAPI | Serving the trained model as a RESTful API |
| Frontend | HTML / CSS / JavaScript | User interface for interacting with the prediction system |
| Deployment | Render | Hosting and Deploying the API and frontend to the web | 

# 📁 Folder Structure  
```
📦 House-Pricing-Prediction  
├── 📁 assets/               # Images generated through EDA and model tuning  

├── 📁 data/                 # Dataset(s) and data resources  

├── 📁 docs/                 # Documents created throughout project, including insights from EDA  

├── 📁 models/               # Trained/serialized ML models (e.g., XGBoost pickle files)  

├── 📁 notebooks/            # Jupyter Notebooks (EDA, modeling experimentation)  

├── 📄 API.py                # Backend API entry point for prediction service  

├── 📄 index.html            # Frontend user interface  

├── 📄 requirements.txt      # Python dependencies  

├── 📄 README.md             # Project overview & instructions  

└── └── __pycache__/         # Python cache folder (auto-generated)  
```

# 🚀 How to test it yourself  
Go to the frontend at this link: https://house-pricing-prediction-xggx.onrender.com. Due to the project's utilisation of Render's free plan, the API spins down after a period of inactivity. As a result, the first usage of the project after a period of inactivity might result in some delays in getting results by the user. This delay is not indicative of a problem with our project, but rather a direct result of using Render's free plan to save on deployment and hosting costs. 

Alternatively, you can also directly test the API through Swagger's UI using this link: https://house-pricing-prediction-dlyo.onrender.com/docs.  


# 🔗 API and Frontend links
API URL: https://house-pricing-prediction-dlyo.onrender.com  
Frontend URL: https://house-pricing-prediction-xggx.onrender.com















