from src.bootstrap_handler import BootstrapHandler
from src.model_handler import ModelHandler
from src.trainpredict_handler import TrainPredictHandler
from functools import partial
from src.data_handler import DataHandler, DataHandlerConfig
from utils.custom_cv import CustomCV


data_handler = DataHandler(DataHandlerConfig())
X, y = data_handler.get_data(['simba'])


class NestedCV:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def run(self):
        # Split data into outer folds
        outer_cv = CustomCV(y=self.y, n_folds=5, random_state=42)

        # Initialize predictions list
        all_fold_predictions = []

        for train_index, val_index in outer_cv.get_indices():

            # Get the data for this outer fold
            X_train_outer, X_val_outer = self.X[train_index], self.X[val_index]
            y_train_outer, y_val_outer = self.y[train_index], self.y[val_index]

            # Perform hyperparameter optimization
            best_params = self.optimize_params()

            # Train main model
            model_handler = ModelHandler(x=X_train_outer, y=y_train_outer)
            model_handler.fit(params=best_params)

            # Perform bootstrapping
            predictions = self.bootstrap(model_handler, X_val_outer, 10)

            # Aggregate predictions
            predictions = self.aggregate_predictions(predictions)

            all_fold_predictions.append(predictions)

        # Get overall predictions
        overall_predictions = self.aggregate_predictions(all_fold_predictions)

    def optimize_params(self):
        ...

    def bootstrap(self, model_handler, X_val, num_bags):
        ...


"""
the outer cross-validation loop is used to split the data into training and validation folds. Within each outer fold, hyperparameter optimization is performed using the inner cross-validation loop. The best hyperparameters are then used to train the final model on the entire outer training set. Bagging is applied to the outer training set by resampling, and multiple models are trained on the bootstrap samples. The predictions from these models are aggregated, and the performance is evaluated on the holdout set.


Split Data into Train/Val/Holdout
  |
  └── Outer Cross-Validation (n_outer_folds)
       |
       ├── Fold 1
       |    |
       |    ├── Inner Cross-Validation (n_inner_folds)
       |    |    |
       |    |    ├── Fold 1.1
       |    |    |    |
       |    |    |    ├── Train Model with Best Hyperparameters
       |    |    |    |
       |    |    |    ├── Bagging with Resampling (n_bootstrap)
       |    |    |    |     |
       |    |    |    |     ├── Bootstrap Model 1
       |    |    |    |     |
       |    |    |    |     ├── Bootstrap Model 2
       |    |    |    |     |
       |    |    |    |     ├── ...
       |    |    |    |     |
       |    |    |    |     └── Bootstrap Model n_bootstrap
       |    |    |    |
       |    |    |    └── Evaluate Performance on Holdout Set
       |    |    |
       |    |    ├── Fold 1.2
       |    |    |    |
       |    |    |    ├── Train Model with Best Hyperparameters
       |    |    |    |
       |    |    |    ├── Bagging with Resampling
       |    |    |    |     |
       |    |    |    |     ├── Bootstrap Model 1
       |    |    |    |     |
       |    |    |    |     ├── Bootstrap Model 2
       |    |    |    |     |
       |    |    |    |     ├── ...
       |    |    |    |     |
       |    |    |    |     └── Bootstrap Model n_bootstrap
       |    |    |    |
       |    |    |    └── Evaluate Performance on Holdout Set
       |    |    |
       |    |    ├── ...
       |    |    |
       |    |    └── Fold n_inner_folds
       |    |          |
       |    |          ├── Train Model with Best Hyperparameters
       |    |          |
       |    |          ├── Bagging with Resampling
       |    |          |     |
       |    |          |     ├── Bootstrap Model 1
       |    |          |     |
       |    |          |     ├── Bootstrap Model 2
       |    |          |     |
       |    |          |     ├── ...
       |    |          |     |
       |    |          |     └── Bootstrap Model n_bootstrap
       |    |          |
       |    |          └── Evaluate Performance on Holdout Set
       |    |
       |    └── Aggregate Predictions from n_inner_folds
       |
       ├── Fold 2
       |    |
       |    ├── Inner Cross-Validation
       |    |    |
       |    |    ├── ...
       |
       └── Fold n_outer_folds
             |
             ├── Inner Cross-Validation
             |    |
             |    ├── ...
             |
             └── Evaluate Performance on Holdout Set (Final Predictions)

                         
Explanation:

1. Data Split: The initial dataset is split into training and validation sets using train_test_split. The training set is further split into outer training and validation sets for cross-validation.

2. Outer Cross-Validation: The outer training set is divided into multiple folds (e.g., Fold 1, Fold 2, Fold 3) using KFold. Each fold is used for training and validation independently.

3. Hyperparameter Optimization: For each fold, hyperparameter optimization is performed (e.g., GridSearchCV) to find the best hyperparameters for that specific fold.

4. Train Model on Fold: The final model is trained on the outer training set using the best hyperparameters determined for that fold.

5. Ensemble Model Predictions: The trained model is used to make predictions on the outer validation set for the specific fold. These predictions are combined to form ensemble predictions.

6. Holdout Set: The holdout set, which was initially split from the data, is used for evaluating the performance of the ensemble predictions.

7. Overall Performance Calculation: The performance metrics (e.g., accuracy, precision) are calculated based on the aggregated ensemble predictions across all folds and compared to the actual values in the holdout set.

This tree-like structure provides a hierarchical representation of the process, highlighting the different stages of data splitting, model training, predictions, and evaluation. It visually demonstrates the flow and dependencies between the steps in a more organized and intuitive manner.






"""

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
from utils.custom_cv import CustomCV

# Split your data into training, validation, and holdout sets
# Split your data into training and holdout sets
X_train_val, X_holdout, y_train_val, y_holdout = train_test_split(
   X, y, test_size=0.2, random_state=42)

# Define the number of folds for outer cross-validation, inner cross-validation, and the number of bootstrap samples for bagging
n_outer_folds = 5
n_inner_folds = 3
n_bootstrap = 10

# Perform nested cross-validation with hyperparameter optimization
outer_cv = KFold(n_splits=n_outer_folds, shuffle=True, random_state=42)

all_fold_predictions = []

for train_index, val_index in outer_cv.split(X_train_val):
    # Get the training and validation sets for the outer fold
    X_train_outer, X_val_outer = X_train_val[train_index], X_train_val[val_index]
    y_train_outer, y_val_outer = y_train_val[train_index], y_train_val[val_index]
    
    # Perform hyperparameter optimization using inner cross-validation
    inner_cv = KFold(n_splits=n_inner_folds, shuffle=True, random_state=42)
    param_grid = {'param1': [value1, value2, ...], 'param2': [value1, value2, ...], ...}
    model = GridSearchCV(ModelClass(), param_grid=param_grid, cv=inner_cv)
    model.fit(X_train_outer, y_train_outer)
    

    # Perform bagging by resampling the outer training set
    bootstrap_samples = []
    for _ in range(n_bootstrap):
        X_bootstrap, y_bootstrap = resample(X_train_outer, y_train_outer, random_state=42)
        bootstrap_samples.append((X_bootstrap, y_bootstrap))


    fold_predictions = []
    # Train a model on each bootstrap sample with the best hyperparameters and make predictions on the validation set
    for X_bs, y_bs in bootstrap_samples:
        model_bs = ModelClass(**model.best_params_)
        model_bs.fit(X_bs, y_bs)
        # Make predictions for this fold 
        y_pred = model_bs.predict(X_val_outer)
        fold_predictions.append(y_pred)
    
    # Aggregate the predictions from multiple models for this fold
    ensemble_predictions = aggregate_predictions(fold_predictions)
    all_fold_predictions.extend(ensemble_predictions)
    
    # Evaluate the performance on the holdout set

    holdout_accuracy = accuracy_score(y_holdout, ensemble_predictions) 

    print(f"Accuracy for this fold on holdout set: {holdout_accuracy}")
    
# Calculate the overall performance metrics using the aggregated predictions from all folds
overall_predictions = aggregate_predictions(all_fold_predictions)
overall_accuracy = accuracy_score(y_holdout, overall_predictions)
print(f"Overall accuracy on holdout set: {overall_accuracy}")









from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.utils import resample
from sklearn.metrics import accuracy_score

# Split your data into training and holdout sets
X_train_val, X_holdout, y_train_val, y_holdout = train_test_split(X, y, test_size=0.2, random_state=42) 

# Define the number of folds for outer cross-validation, inner cross-validation, and the number of bootstrap samples for bagging
n_outer_folds = 5
n_inner_folds = 3   
n_bootstrap = 10

# Perform nested cross-validation with hyperparameter optimization
outer_cv = KFold(n_splits=n_outer_folds, shuffle=True, random_state=42)

all_fold_predictions = []

for train_index, val_index in outer_cv.split(X_train_val):
    # Get the data for this outer fold  
    X_train_outer, X_val_outer = X_train_val[train_index], X_train_val[val_index]   
    y_train_outer, y_val_outer = y_train_val[train_index], y_train_val[val_index]  
      
    # Perform hyperparameter optimization     
    model = GridSearchCV(ModelClass(), param_grid=param_grid, cv=inner_cv)     
    model.fit(X_train_outer, y_train_outer)
    
    # Perform bagging          
    for X_bs, y_bs in bootstrap_samples:
        model_bs = ModelClass(**model.best_params_)
        model_bs.fit(X_bs, y_bs)     
   
       # Make predictions for this fold        
        y_pred = model_bs.predict(X_val_outer)



"""
Here are some details about the `aggregate_predictions()` function:

- It takes a list of predictions (`fold_predictions`) as input. This list contains the predictions from each model in the bagging ensemble for that outer fold.

- It then aggregates these predictions into a single prediction for that fold. There are a few common ways to do this:

1. Take the mean - Simply average all the predictions. This is a simple and effective approach.

2. Take the median - Take the median of all the predictions. Less sensitive to outliers than the mean.

3. Weighted average - Weight the predictions by each model's accuracy and take the weighted average. Models with higher accuracy are weighted more.

4. Voting - Have each model "vote" for a class/value, and take the class/value with the most votes.

In this case, we implement the mean approach:

```python
def aggregate_predictions(fold_predictions):
    ensemble_predictions = np.mean(fold_predictions, axis=0)
    return ensemble_predictions
```

- This takes the mean of all the predictions in the `fold_predictions` list, along the first axis (which corresponds to the individual models).

- The result, `ensemble_predictions`, is a single array of predictions for that outer fold, by aggregating the predictions from all the models in the bagging ensemble.

- This ensemble prediction is then used to calculate the holdout accuracy for that fold, and eventually all ensemble predictions are aggregated to calculate the overall accuracy.

So in summary, the `aggregate_predictions()` function allows us to aggregate the predictions from multiple models (the bagging ensemble) into a single ensemble prediction, which we can then evaluate.

Hope this helps explain the `aggregate_predictions()` function! Let me know if you have any other questions.

"""
