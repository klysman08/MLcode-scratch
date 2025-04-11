# Machine Learning Algorithms Implementation

This project contains implementations of various fundamental machine learning algorithms from scratch using Python and NumPy. The implementations are designed to be educational and demonstrate the core concepts behind these algorithms.

## Implemented Algorithms

### 1. Linear Regression
- Simple linear regression implementation
- Coefficient estimation
- R² score calculation
- Located in `class_LinearRegrassion.py`

### 2. K-Nearest Neighbors (KNN)
- KNN classification algorithm
- Euclidean distance-based classification
- Visualization capabilities
- Located in `class_Kmeans.py`

### 3. Multiple Regression
- Multiple variable regression implementation
- Located in `class_MultipleRegression.py`

### 4. Polynomial Regression
- Polynomial regression implementation
- Located in `class_PolynomilRegression.py`

### 5. Logistic Regression
- Binary classification using logistic regression
- Located in `class_LogisticRegression.py`

## Project Structure

```
.
├── datasets/                  # Directory for datasets
├── class_LinearRegrassion.py  # Linear regression implementation
├── class_Kmeans.py           # KNN implementation
├── class_MultipleRegression.py # Multiple regression implementation
├── class_PolynomilRegression.py # Polynomial regression implementation
├── class_LogisticRegression.py # Logistic regression implementation
├── class_numpy.py            # NumPy utility functions
├── pyproject.toml            # Project dependencies
└── README.md                 # This file
```

## Requirements

- Python 3.x
- NumPy
- SciPy
- Matplotlib

## Environment Setup

This project uses `uv` for dependency management. To set up the environment:

1. Install `uv` if you haven't already:
```bash
pip install uv
```

2. Create and activate a virtual environment:
```bash
uv venv
.venv\Scripts\activate  # On Windows
source .venv/bin/activate  # On Unix/MacOS
```

3. Install dependencies:
```bash
uv pip install -r requirements.txt
```

## Usage

Each algorithm is implemented as a class with standard methods:
- `fit()` or `estimate_coef()` for training
- `predict()` for making predictions
- Additional utility methods specific to each algorithm

Example usage for Linear Regression:
```python
from class_LinearRegrassion import LinearRegression

# Create model instance
model = LinearRegression()

# Prepare data
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])

# Train model
b0, b1 = model.estimate_coef(x, y)

# Get R² score
r2 = model.R2(x, y)
```

## Educational Purpose

This project is primarily designed for educational purposes to:
- Understand the mathematical foundations of machine learning algorithms
- Learn how to implement ML algorithms from scratch
- Practice good coding practices in Python
- Gain hands-on experience with NumPy and scientific computing

## Contributing

Feel free to contribute to this project by:
- Adding new algorithms
- Improving existing implementations
- Adding more documentation
- Fixing bugs
- Adding test cases

## License

This project is open source and available for educational purposes.


