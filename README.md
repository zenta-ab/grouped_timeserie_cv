# Grouped Timeseries Crossvalidation

By integrating pipelines, parameter selection, cross-validation, and model evaluation, we have developed algorithms that efficiently identify the optimal model for event-based timeserie data. This approach automates trial-and-error tasks, significantly saving developers time. The use of Scikit-learn's robust tools ensures a streamlined and efficient process, enhancing model performance even in data-constrained environments.


## Pipelines Definition
First, we define our pipelines. Here, we have chosen three classification models: Gaussian Naive Bayes, Decision Tree, and Logistic Regression. However, you can test other options such as Support Vector Machines, Neural Networks, XGBoost, or Random Forests. Each pipeline includes steps for scaling, feature selection, and model fitting, with a variety of parameters to tune.

```python
pipelines = [
    Pipeline([
        ('scaler', StandardScaler()),
        ('selector', SelectKBest()),
        ('model', GaussianNB())
    ]),
    Pipeline([
        ('scaler', StandardScaler()),
        ('selector', SelectKBest()),
        ('model', DecisionTreeClassifier())
    ]),
    Pipeline([
        ('scaler', StandardScaler()),
        ('selector', SelectKBest()),
        ('model', LogisticRegression())
    ])
]
```
## Parameter Grids
The following parameter grids define the hyperparameters for each of the pipelines. The grids are organized to match the classifiers used in the pipelines.

```python

param_grids = [
    # Grid for Gaussian Naive Bayes
    {
        'scaler__with_mean': [True, False],
        'scaler__with_std': [True, False],
        'selector__k': [3, 5, 'all'],
        'selector__score_func': [mutual_info_classif],
        'model__var_smoothing': [1e-9, 1e-8, 1e-7]
    },
    # Grid for Decision Tree Classifier
    {
        'scaler__with_mean': [True, False],
        'scaler__with_std': [True, False],
        'selector__k': [3, 5, 'all'],
        'selector__score_func': [mutual_info_classif],
        'model__criterion': ['gini', 'entropy'],
        'model__splitter': ['best', 'random'],
        'model__max_depth': [None, 1, 2, 5, 10],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4],
        'model__min_weight_fraction_leaf': [0.0, 0.1, 0.2],
        'model__max_features': [None, 'auto', 'sqrt', 'log2'],
        'model__random_state': [42]
    },
    # Grid for Logistic Regression
    {
        'scaler__with_mean': [True, False],
        'scaler__with_std': [True, False],
        'selector__k': [3, 5, 'all'],
        'selector__score_func': [mutual_info_classif],
        'model__C': [1e-12, 0.1, 1, 10],
        'model__penalty': ['l2', 'l1'],
        'model__solver': ['liblinear'],
        'model__max_iter': [1000],
        'model__class_weight': ['balanced']
    }
]

```

## Load the Dataset
Load the dataset from a CSV file and convert the 'DateTime' column to a datetime object.
```python
data = pd.read_csv('model_data.csv')
data['DateTime'] = pd.to_datetime(data['DateTime'])
```

## Classification
Perform classification using grouped time series cross-validation with the defined pipelines and parameter grids.

```python
from grouped_timeserie_cv import GroupedTimeSerieCV
grouped_cv = GroupedTimeSerieCV()
result = grouped_cv.classify(data, pipelines, param_grids, 'D', 'DateTime','Label', 'accuracy')
```
Where the optional parameters are the frequency, DateTime column, label column, and scoring method.


## Expected Output

During training, you will see output similar to the following in the console:

```
Process model: GaussianNB
Score: 0.7813909774436091
Process model: DecisionTreeClassifier
Score: 0.8111528822055137
Process model: LogisticRegression
Score:  0.8364661654135338
Best model: LogisticRegression
Best parameters for best results: {
 'model__C': 1,
 'model__class_weight': 'balanced',
 'model__max_iter': 1000,
 'model__penalty': 'l2',
 'model__solver': 'liblinear',
 'scaler__with_mean': True,
 'scaler__with_std': True,
 'selector__k': 3
}
Selected features: ['Moisture', 'Temperature', 'MeanTemperaturePeak']

```

In this example, Logistic Regression is the best model with an accuracy of 83.6%.

## Define Result Data Class

The `CrossValidationResult` class holds the results from the cross-validation algorithm.

```python
class CrossValidationResult:
    confusion_matrices: np.ndarray
    class_labels: list
    train_sizes: np.ndarray
    train_mean: np.ndarray
    train_std: np.ndarray
    test_mean: np.ndarray
    test_std: np.ndarray
    best_model: object
    selected_feature_names: list
    best_params: dict
    incorrect_dates: np.ndarray
    actual_values: np.ndarray
    predicted_values: np.ndarray
```

## Plot Results

Use the results to plot a confusion matrix and learning curve by accessing the plotter instance .

```python
# Example plotting functions
grouped_cv.plotter.plot_confusion_matrix(result.confusion_matrices, result.class_labels)
grouped_cv.plotter.plot_learning_curve(result.train_sizes, result.train_mean, result.train_std, result.test_mean, result.test_std)
```

