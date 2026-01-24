# AAPL Stock Price Prediction - Multi-Framework ML Project

A comprehensive machine learning project comparing multiple frameworks (scikit-learn, TensorFlow) for predicting Apple (AAPL) stock prices using technical indicators and deep learning.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.0-orange)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 🎯 Project Overview

This project demonstrates end-to-end machine learning workflow for stock price prediction, featuring:
- **18 technical indicators** (Moving Averages, RSI, MACD, Bollinger Bands, etc.)
- **4 ML models** compared side-by-side
- **Professional visualizations** and performance metrics
- **Modular, production-ready code** structure

## 📊 Models Implemented

### Traditional ML (scikit-learn)
1. **Linear Regression** - Baseline model
2. **Random Forest Regressor** - Ensemble learning
3. **Support Vector Regression (SVR)** - Non-linear predictions

### Deep Learning (TensorFlow/Keras)
4. **Neural Network** - Multi-layer feed-forward network with dropout

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/AAPL-stock-prediction.git
cd AAPL-stock-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Project
```bash
python main.py
```

This will:
1. Download/load AAPL stock data (2020-2024)
2. Generate 18 technical indicators
3. Train all 4 models
4. Evaluate and compare performance
5. Generate visualizations in `results/` folder

## 📁 Project Structure
```
AAPL-stock-prediction/
├── data/                    # Data storage
├── models/                  # ML model implementations
│   ├── linear_regression.py # scikit-learn models
│   ├── neural_network.py    # TensorFlow model
│   └── model_comparison.py  # Model comparison framework
├── utils/                   # Utility functions
│   ├── data_loader.py       # Data loading & feature engineering
│   ├── evaluation.py        # Metrics & visualization
│   └── sample_data.py       # Sample data generator
├── results/                 # Generated visualizations
├── main.py                  # Main execution script
├── requirements.txt         # Dependencies
└── README.md               # Project documentation
```

## 📈 Features

### Technical Indicators
- **Moving Averages**: MA(5, 10, 20, 50), EMA(12, 26)
- **Momentum Indicators**: RSI, MACD, Price Momentum
- **Volatility**: Bollinger Bands, Historical Volatility
- **Volume Analysis**: Volume Moving Averages

### Model Evaluation Metrics
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score
- Mean Absolute Percentage Error (MAPE)

### Visualizations
- Actual vs Predicted price plots
- Residual analysis
- Model comparison charts

## 🛠️ Technologies Used

- **Python 3.11**
- **scikit-learn** - Traditional ML algorithms
- **TensorFlow/Keras** - Deep learning framework
- **pandas** - Data manipulation
- **NumPy** - Numerical computing
- **matplotlib/seaborn** - Data visualization
- **yfinance** - Stock data retrieval

## 📊 Results

The project evaluates all models on test data and identifies the best performer based on RMSE. Results are saved in the `results/` directory with:
- Individual model prediction plots
- Residual analysis
- Comprehensive model comparison

## 🎥 Demo

Here’s a quick look at some of the sample results:

<p align="center">
  <img src="https://i.imgur.com/SwALQDI.png" alt="Image 1" width="600"/>
</p>

<p align="center">
  <img src="https://i.imgur.com/Ir10AJM.png" alt="Image 2" width="600"/>
</p>

<p align="center">
  <img src="https://i.imgur.com/Lz7SQ5y.png" alt="Image 3" width="600"/>
</p>

<p align="center">
  <img src="https://i.imgur.com/jwBzBj1.png" alt="Image 4" width="600"/>
</p>

## 🔧 Customization

### Change Stock Ticker
```python
loader = StockDataLoader(ticker="MSFT", start_date="2020-01-01")
```

### Adjust Date Range
```python
loader = StockDataLoader(ticker="AAPL", start_date="2022-01-01", end_date="2024-12-31")
```

### Modify Neural Network Architecture
```python
nn_model.build_model(architecture='deep')  # Options: 'standard', 'deep', 'wide'
```

## 📝 Key Learnings

- Feature engineering with technical indicators significantly improves predictions
- Neural networks can capture non-linear patterns but require more data
- Ensemble methods (Random Forest) provide robust predictions
- Proper train/test splitting is crucial for time series data

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests

## 📄 License

This project is licensed under the MIT License.

## 👤 Author

**Miguel Shema Ngabonziza**
- LinkedIn: [linkedin.com/in/migztech](https://linkedin.com/in/migztech)
- GitHub: [github.com/SNMiguel](https://github.com/SNMiguel)
- Portfolio: [migztech.vercel.app](https://migztech.vercel.app)

## 🙏 Acknowledgments

- Data provided by Yahoo Finance via yfinance
- Inspired by quantitative finance and ML best practices
- Built as part of technical skill development for ML/AI roles

---

⭐ If you found this project helpful, please consider giving it a star!