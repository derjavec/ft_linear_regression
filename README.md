# Car Price Prediction with Linear Regression


## Overview

This project implements a **simple linear regression model** to predict car prices based on mileage (`km`).  
It is built from scratch using **NumPy and Pandas**, without relying on machine learning libraries like `scikit-learn`.  

The model is trained using **gradient descent**, and the code includes functionality to:

- Clean and preprocess the dataset
- Scale and descale input and output values
- Train the linear regression model
- Evaluate the model for different learning rates (alpha)
- Visualize predictions and compare them to real data
- Select the best learning rate based on prediction accuracy

---

## Features

- **Custom Linear Regression:** Fully implemented gradient descent with feature scaling.
- **Multiple Alpha Evaluation:** Automatically tests multiple learning rates and selects the one with the best accuracy.
- **Precision Metrics:** Calculates the R² score for regression and provides a clear precision measure between 0 and 1.
- **CSV Output:** Generates a CSV file with predicted prices for different alpha values.
- **Visualization:** Plots predictions versus actual prices and highlights the best model.

---

## Project Structure

