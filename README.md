# Stock Price Prediction - Multi-Framework ML Project

A comprehensive machine learning project demonstrating end-to-end ML pipeline development by comparing multiple frameworks (scikit-learn, TensorFlow) for stock price prediction using advanced technical indicators and deep learning.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.0-orange)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 🎯 Project Overview

This project showcases a complete machine learning workflow for stock price prediction, featuring:
- **18 engineered technical indicators** (Moving Averages, RSI, MACD, Bollinger Bands, Momentum)
- **4 ML models** trained and compared side-by-side
- **Multiple frameworks** (scikit-learn + TensorFlow) in a single project
- **Professional visualizations** and comprehensive performance metrics
- **Modular, production-ready code** structure

**Use Case:** Predicting Apple (AAPL) stock closing prices using historical data and technical analysis.

## 📊 Models Implemented

### Traditional ML (scikit-learn)
1. **Linear Regression** - Baseline linear model
2. **Random Forest Regressor** - Ensemble learning approach
3. **Support Vector Regression (SVR)** - Non-linear kernel-based predictions

### Deep Learning (TensorFlow/Keras)
4. **Neural Network** - Multi-layer feed-forward architecture with dropout regularization

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/SNMiguel/Stock-Price-Prediction-ML.git
cd Stock-Price-Prediction-ML

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

**What happens:**
1. Downloads/loads AAPL stock data (2020-2024)
2. Engineers 18 technical indicators from raw OHLCV data
3. Trains all 4 models with proper train/test splitting
4. Evaluates and compares model performance
5. Generates visualizations saved to `results/` folder

## 📁 Project Structure
```
Stock-Price-Prediction-ML/
├── data/                    # Data storage (cached/sample data)
├── models/                  # ML model implementations
│   ├── linear_regression.py # scikit-learn models (LR, RF, SVR)
│   ├── neural_network.py    # TensorFlow/Keras deep learning model
│   └── model_comparison.py  # Framework for comparing all models
├── utils/                   # Utility functions
│   ├── data_loader.py       # Data loading & feature engineering
│   ├── evaluation.py        # Metrics calculation & visualization
│   └── sample_data.py       # Sample data generator (fallback)
├── results/                 # Generated visualizations & plots
├── main.py                  # Main execution script
├── requirements.txt         # Project dependencies
└── README.md               # Project documentation
```

## 📈 Features

### Feature Engineering - Technical Indicators
- **Moving Averages**: Simple MA (5, 10, 20, 50-day), Exponential MA (12, 26-day)
- **Momentum Indicators**: RSI (14-day), MACD, Price Momentum (5, 10-day)
- **Volatility Measures**: Bollinger Bands (20-day), Historical Volatility
- **Volume Analysis**: Volume Moving Averages (5, 20-day)
- **Returns**: Daily percentage returns

### Model Evaluation Metrics
- **Mean Absolute Error (MAE)** - Average prediction error
- **Root Mean Squared Error (RMSE)** - Penalizes large errors
- **R² Score** - Proportion of variance explained
- **Mean Absolute Percentage Error (MAPE)** - Relative error percentage

### Visualizations
- Actual vs Predicted price time series plots
- Residual distribution and scatter plots
- Side-by-side model comparison charts
- Performance metrics dashboard

## 🛠️ Technologies & Frameworks

- **Python 3.11** - Programming language
- **scikit-learn 1.4.0** - Traditional ML algorithms (Linear Regression, Random Forest, SVR)
- **TensorFlow 2.15.0** - Deep learning framework (Keras API)
- **pandas** - Data manipulation and time series handling
- **NumPy** - Numerical computing and array operations
- **matplotlib/seaborn** - Data visualization and plotting
- **yfinance** - Stock market data retrieval

## 📊 Sample Results

The project evaluates all models on held-out test data and identifies the best performer based on RMSE. Results include:
- Individual model prediction plots with actual vs predicted prices
- Residual analysis showing prediction error distribution
- Comprehensive side-by-side model comparison

### Visualizations

<p align="center">
  <img src="https://i.imgur.com/SwALQDI.png" alt="Model Predictions" width="600"/>
</p>

<p align="center">
  <img src="https://i.imgur.com/Ir10AJM.png" alt="Residual Analysis" width="600"/>
</p>

<p align="center">
  <img src="https://i.imgur.com/Lz7SQ5y.png" alt="Model Comparison" width="600"/>
</p>

<p align="center">
  <img src="https://i.imgur.com/jwBzBj1.png" alt="Performance Metrics" width="600"/>
</p>

## 🔧 Customization

### Change Stock Ticker
```python
# In main.py
loader = StockDataLoader(ticker="MSFT", start_date="2020-01-01")
```

### Adjust Date Range
```python
loader = StockDataLoader(
    ticker="AAPL", 
    start_date="2022-01-01", 
    end_date="2024-12-31"
)
```

### Modify Neural Network Architecture
```python
# Options: 'standard', 'deep', 'wide'
nn_model.build_model(architecture='deep')
```

### Add More Models
The modular structure makes it easy to add new models:
```python
# In models/linear_regression.py
def train_custom_model(self, X_train, y_train):
    model = YourCustomModel()
    model.fit(X_train, y_train)
    self.models['Custom Model'] = model
    return model
```

## 📝 Key Learnings & Insights

- **Feature engineering** with domain-specific technical indicators significantly improves prediction accuracy over raw OHLCV data
- **Neural networks** can capture complex non-linear patterns but require careful tuning and sufficient data
- **Ensemble methods** like Random Forest provide robust predictions with built-in feature importance
- **Proper train/test splitting** without shuffling is critical for time series to avoid data leakage
- **Model comparison** reveals that no single model dominates across all metrics

## 🎓 Skills Demonstrated

- End-to-end ML pipeline development
- Feature engineering for time series data
- Multiple ML framework integration (scikit-learn, TensorFlow)
- Model training, evaluation, and comparison
- Data visualization and results communication
- Clean, modular code architecture
- Git version control and documentation

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to:
- Report bugs or issues
- Suggest new features or models
- Submit pull requests
- Improve documentation

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Miguel Shema Ngabonziza**
- LinkedIn: [linkedin.com/in/migztech](https://linkedin.com/in/migztech)
- GitHub: [github.com/SNMiguel](https://github.com/SNMiguel)
- Portfolio: [migztech.vercel.app](https://migztech.vercel.app)

## 🙏 Acknowledgments

- Stock data provided by Yahoo Finance via the yfinance library
- Inspired by quantitative finance and machine learning best practices
- Built as part of technical skill development for AI/ML engineering roles

## 🔮 Future Enhancements

- [ ] Add LSTM/GRU models for sequential pattern learning
- [ ] Implement sliding window predictions for multi-day forecasting
- [ ] Create interactive Streamlit/Gradio web interface
- [ ] Add support for multiple stock tickers simultaneously
- [ ] Implement hyperparameter tuning with GridSearchCV
- [ ] Deploy model as REST API with FastAPI

---

⭐ **If you found this project helpful, please consider giving it a star!**