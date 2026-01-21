# 📈 Stock Price Prediction System using LSTM Neural Networks

This project is an interactive web-based machine learning application that predicts stock prices using Long Short-Term Memory (LSTM) neural networks and real-time financial data.

This is my first end-to-end AI/ML project where I implemented data preprocessing, deep learning modeling, and web deployment in a single pipeline.

---

## 🚀 Features

- Fetches real-time and historical stock data using Yahoo Finance (yfinance)  
- Supports multiple NSE stocks with dynamic stock selection  
- Preprocesses time-series data using scaling and sliding window techniques  
- LSTM-based deep learning model for stock price prediction  
- Interactive web interface built using Streamlit  
- Visualizes actual vs predicted prices  
- Forecasts future stock prices  

---

## 🧠 Tech Stack

- Python  
- TensorFlow / Keras  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib  
- Streamlit  
- yfinance  

---

## 📂 Project Structure

Stock-Price-Prediction-LSTM/
│
├── app.py # Streamlit application
├── model.h5 # Trained LSTM model
├── requirements.txt # Project dependencies
├── README.md # Project documentation

---

## ▶️ How to Run the Project

1. Clone the repository:  
git clone https://github.com/AnselSharma/Stock-Price-Prediction-LSTM.git

2. Create and activate virtual environment: (IMP)

python -m venv lstm_env

lstm_env\Scripts\activate

3. Install dependencies:
pip install numpy pandas matplotlib scikit-learn yfinance streamlit tensorflow

4. Run the Streamlit app:

streamlit run app.py

---

## 🌐 Live Demo

👉 http://10.173.7.54:8501

---

## 🧠 Model Architecture

The prediction model is based on a Long Short-Term Memory (LSTM) neural network designed for time-series forecasting.

- Input Sequence Length: 100 days  
- LSTM Layers for temporal feature extraction  
- Dense output layer for price prediction  
- Optimizer: Adam  
- Loss Function: Mean Squared Error (MSE)  

The model is trained on historical closing prices to capture long-term dependencies in stock price movements.


---

## 📂 Dataset & Data Source

- Stock market data is fetched dynamically using the **Yahoo Finance API (yfinance)**  
- Includes historical OHLC (Open, High, Low, Close) price data  
- Data is preprocessed using normalization and sliding window techniques  
- Supports multiple NSE-listed stocks  

The dataset is generated in real-time based on the selected stock and date range.


---

## 📊 Model Evaluation

The model performance is evaluated using:

- Mean Squared Error (MSE)  
- Root Mean Squared Error (RMSE)  
- Visual comparison of actual vs predicted stock prices  

Prediction accuracy is analyzed through graphical visualization of test data.


---

## 🚀 Future Improvements

- Add support for more stock exchanges (BSE, NASDAQ, NYSE)  
- Implement multi-step future forecasting  
- Integrate technical indicators (RSI, MACD, Moving Averages)  
- Add model retraining option from the web interface  
- Improve prediction accuracy using GRU or Transformer models  
- Deploy the application on cloud platforms (Streamlit Cloud / AWS)  


---

## ⚠️ Limitations

- Stock price prediction is inherently uncertain due to market volatility  
- Model performance depends on historical data quality  
- External factors such as news and economic events are not considered  
- The model is designed for educational and research purposes only  


---

## 📄 License

This project is intended for educational and research purposes.  
You are free to use, modify, and distribute this project with proper credit.
