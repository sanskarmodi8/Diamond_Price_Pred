# Basic Import
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, make_scorer
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.neural_network import MLPRegressor
from src.utils import save_object
from dataclasses import dataclass
import sys
import os


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_training(self, train_array, test_array):
        try:
            logging.info('Model Training Initiated')
            logging.info(
                'Splitting Dependent and Independent variables from train and test data')
            X_train_full, y_train_full, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            X_train, X_valid, y_train, y_valid = train_test_split(
                X_train_full, y_train_full, test_size=0.2, random_state=42)

            # Define a dictionary of regression metrics
            regression_metrics = {
                "MSE": mean_squared_error,
                "MAE": mean_absolute_error,
                "RMSE": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
                "R-squared": r2_score
            }

            # Initialize variables to store the best model and its performance
            best_model = None
            best_models = []
            best_scores = {metric_name: -np.inf for metric_name in regression_metrics}


            # Create lists to store base models (best_estimator for each regressor)
            base_models = []


            # hyperparameter grids
            ridge_param_grid = {
                'alpha': [0.01, 0.1, 1.0],
            }

            lasso_param_grid = {
                'alpha': [0.01, 0.1, 1.0],
            }

            dt_param_grid = {
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
            }

            rf_param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
            }

            gb_param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 4, 5],
            }

            svr_param_grid = {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['linear', 'rbf', 'poly'],
                'degree': [2, 3, 4],
            }

            xgb_param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 4, 5],
                'learning_rate': [0.01, 0.1, 0.2],
            }

            mlp_param_grid = {
                'hidden_layer_sizes': [(32, 16), (64, 32), (128, 64), (64, 32, 16)],
                'epochs': [50, 100, 200],
                'batch_size': [32, 64, 128],
            }

            # Define a list of regression algorithms to try, including the custom MLP
            regressors = [
                ("Linear Regression", LinearRegression(), None),
                ("Ridge Regression", Ridge(), ridge_param_grid),
                ("Lasso Regression", Lasso(), lasso_param_grid),
                ("Decision Tree", DecisionTreeRegressor(), dt_param_grid),
                ("Random Forest", RandomForestRegressor(), rf_param_grid),
                ("Gradient Boosting", GradientBoostingRegressor(), gb_param_grid),
                ("Support Vector Machine", SVR(), svr_param_grid),
                ("XGBoost", XGBRegressor(), xgb_param_grid),
                # Placeholder for Keras model
                ("Neural Network (Custom MLP)", None, mlp_param_grid)
            ]

            base_model_scores = {model_name: [] for model_name, _, _ in regressors}
            best_models_score = {model_name: [] for model_name, _, _ in regressors}


            # Function to perform model training and hyperparameter tuning

            def train_model(name, model, param_grid):
                mse = None

                if param_grid is not None:
                    
                    if name == "Neural Network (Custom MLP)":
                        
                        # Define a custom scorer for negative mean squared error
                        scorer = make_scorer(lambda y_true, y_pred: -mean_squared_error(y_true, y_pred))

                        # Create the MLPRegressor model
                        mlp_regressor = MLPRegressor(random_state=42)

                        # Use RandomizedSearchCV for hyperparameter tuning
                        search = RandomizedSearchCV(
                            estimator=mlp_regressor,
                            param_distributions=param_grid,
                            cv=5,
                            scoring=scorer,  # Use the custom scorer for negative MSE
                            n_jobs=-1,
                            n_iter=10,
                            random_state=42,
                        )

                        # Fit the search to your data (X_train and y_train)
                        search.fit(X_train, y_train)

                        # Get the best estimator
                        best_estimator = search.best_estimator_
                        mse = -search.best_score_
                        y_pred = best_estimator.predict(X_test)

                        
                        for metric_name, metric_func in regression_metrics.items():
                            metric_score = metric_func(y_test, y_pred)
                            print(f"{name} - {metric_name}: {metric_score}")

                            base_model_scores[name].append(
                                f"{metric_name}: {metric_score}")
                        print("Best Params: ")
                        print(search.best_params_)
                        print("\n")
                        base_models.append(('MLP', best_estimator))

                    else:
                        grid_search = GridSearchCV(
                            estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
                        grid_search.fit(X_train, y_train)

                        best_estimator = grid_search.best_estimator_
                        mse = -grid_search.best_score_
                        y_pred = best_estimator.predict(X_test)

                        
                        for metric_name, metric_func in regression_metrics.items():
                            metric_score = metric_func(y_test, y_pred)
                            print(f"{name} - {metric_name}: {metric_score}")
                            base_model_scores[name].append(
                                f"{metric_name}: {metric_score}")
                        print("Best Params: ")
                        print(grid_search.best_params_)
                        print("\n")
                        base_models.append((name, best_estimator))
                                
                    
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    for metric_name, metric_func in regression_metrics.items():
                        metric_score = metric_func(y_test, y_pred)
                        print(f"{name} - {metric_name}: {metric_score}")
                        base_model_scores[name].append(
                            f"{metric_name}: {metric_score}")
                    print("\n")

                    base_models.append((name, model))

                if mse is not None and mse > best_scores['MSE']:
                    best_scores['MSE'] = mse
                    best_model = best_estimator
                    best_models.append((name,best_model))
                    for metric_name, metric_func in regression_metrics.items():
                        metric_score = metric_func(y_test, y_pred)
                        best_models_score[name].append(
                            f"{metric_name}: {metric_score}")
                    print("\n")

                

            for name, model, param_grid in regressors:
                logging.info('Training model: ' + name)
                train_model(name, model, param_grid)
                logging.info('Training completed for model: ' + name)
                
            logging.info('Training Stack Model')
            # Create a StackingRegressor with the base models and a meta-model (Linear Regression)
            stacking_regressor = StackingRegressor(
                estimators=best_models, final_estimator=LinearRegression())

            # Train the StackingRegressor on the training data
            stacking_regressor.fit(X_train, y_train)
            
            logging.info('Training completed for Stack Model')

            # Make predictions using the stacked model
            y_pred_stacked = stacking_regressor.predict(X_test)

            # Calculate and print regression metric scores for the stacked model
            mse_stacked = None  # Initialize mse_stacked outside the loop
            for metric_name, metric_func in regression_metrics.items():
                metric_score = metric_func(y_test, y_pred_stacked)
                if metric_name == "MSE":
                    mse_stacked = metric_score  # Assign mse_stacked for later comparison
                print(f"Stacked Model - {metric_name}: {metric_score}")
            print("\n")

            # Compare mse_stacked to the best MSE
            if mse_stacked is not None and mse_stacked > best_scores['MSE']:
                best_scores['MSE'] = mse_stacked  # Update best_scores with the new MSE
                best_model = stacking_regressor
                best_models.append(('Stacking Regressor',best_model))  # Append the stacked model
                for metric_name, metric_func in regression_metrics.items():
                    metric_score = metric_func(y_test, y_pred_stacked)
                    best_models_score['Stacking Regressor'].append(
                        f"{metric_name}: {metric_score}")


            print(f"Best Models: \n{ best_models }")
            print(f"Base Models: \n{ base_models }")
            print(f"Base Model Scores: \n{ base_model_scores }")
            print(f"Best Model Scores: \n{ best_models_score }")
            print(f"Found the best model!  --> { best_model }")

            logging.info('Training results:')
            logging.info(f"Best Models: \n{ best_models }")
            logging.info(f"Base Models: \n{ base_models }")
            logging.info(f"Base Model Scores: \n{ base_model_scores }")
            logging.info(f"Best Model Scores: \n{ best_models_score }")
            logging.info(f"Found the best model!  --> { best_model }")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info('Model pickle created and saved')
            logging.info('Model Training Completed')

        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e, sys)
