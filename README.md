# Stock Market Price Predictor

This project is a stock market price predictor built using Machine Learning and various Python libraries. The goal is to predict future stock prices based on historical data. This project is developed by Roshan Panta as the final project for the BCA 8th semester.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Installation

To get started, clone the repository and install the required dependencies.

```bash
git clone https://github.com/Roshan2059/stock-market-price-predictor.git
First make a virtual environment using: python -m venv env
activate  the virtual env:env\scripts\activate
cd stock-market-price-predictor
pip install -r requirements.txt
```

## Usage

Open the `stock-price.ipynb` notebook and follow the instructions to load the data, train the model, and make predictions.

1. Start Jupyter Notebook:
2. Open `stock-price.ipynb` in the Jupyter Notebook interface.
3. Follow the steps in the notebook to load the data, preprocess it, train the model, and make predictions.

## Project Structure

- `requirements.txt`: A file containing the list of dependencies required for the project.
- `stock-price.ipynb`: The Jupyter notebook containing the code for data processing, model training, and prediction.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE]() file for more details.

usecaseDiagram
actor User
actor System

    User --> (Enter Stock Symbol)
    User --> (Select Model for Prediction)
    User --> (Select Date for Prediction)
    User --> (View Stock Data)
    User --> (Predict Next 7 Days)
    User --> (View Actual vs Predicted Data)
    User --> (View Future Predictions)

    (Enter Stock Symbol) --> (View Stock Data)
    (Select Model for Prediction) --> (View Actual vs Predicted Data)
    (Select Date for Prediction) --> (View Stock Data)
    (Predict Next 7 Days) --> (View Future Predictions)
    (View Stock Data) --> System
    (View Actual vs Predicted Data) --> System
    (View Future Predictions) --> System

    System --> (Fetch Stock Data)
    System --> (Preprocess Data)
    System --> (Train Model: LSTM or Regression)
    System --> (Make Predictions)
    System --> (Visualize Data)

    System --> (Store/Load Model)
