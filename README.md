# COMP379_March_Madness_Predictor

## Addressed Problem Details and Motivation 

The project aims to utilize machine learning techniques to predict the outcome of the NCAA March Madness basketball games. With data spanning the last 15 years of March Madness statistics, the goal is to develop a model capable of accurately predicting game winners. The motivation behind this project was each of our interests and the immense popularity surrounding March Madness. Predicting game outcomes could have significant implications for basketball fans, bettors, and sports analysts. 
  

## Formal Problem Definition 

**Task (T):** Predict the winning percentage of March Madness teams 

**Experience (E):** Data from March Madness seasons 2002-2019 

https://www.kaggle.com/datasets/nishaanamin/march-madness-data 

**Performance (P):** Accuracy in predicting a team’s winning percentage 

## Metrics 

### Evaluation Metrics:  

- Mean Squared Error (MSE) 

- R Squared (R^2) 

- Mean Absolute Error (MAE) 

### Models 

- Linear Regression (With Stepwise Regression) 

  - Train RMSE: 0.059534579738275605 

  - Test RMSE: 0.0839783171013451 

  - Train R^2: 0.9354877866832415 

  - Test R^2: 0.8719361629312715 

- SVM for Regression (With Stepwise Regression) 

  - Train MSE: 0.005071586635460595  

  - Test MSE: 0.007923909728489692  

  - Train R^2: 0.8759123857741552  

  - Test R^2: 0.8033526510106256  

  - Train MAE: 0.0623489868371097  

  - Test MAE: 0.06996411135115455 

- Random Forest for Regression (With Stepwise Regression) 

  - Train MSE:  1.0108503613181397e-05 

  - Test MSE:  5.605191853239837e-05 

  - Train R^2:  0.999815141844911 

  - Test R^2:  0.9989718253667119 

  - Train MAE:  0.0012113874138681877 

  - Test MAE:  0.0028673426765913903 

- Neural Network 

  - Train MSE: 0.004650577991182201  

  - Test MSE: 0.009215902974294038  

  - Train R^2: 0.912418176436343  

  - Test R^2: 0.8604091569472837  

  - Train MAE: 0.04788041231369084  

  - Test MAE: 0.06552074742953605 

## Chosen ML Models 

- Support Vector Machine (SVM) for regression 

- Random Forest for regression 

- Linear Regression 

- Neural Network 

## Training and Test Details 

- Data is generated synthetically with 1000 samples and 10 features 

- Data is split into training and testing sets with a size of 20% 
 
