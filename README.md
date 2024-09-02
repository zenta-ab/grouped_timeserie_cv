
## 


# Pipelines Definition
The following code snippet defines three distinct machine learning pipelines, each with different classifiers. Each pipeline consists of a data scaler, a feature selector, and a model.

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

# Parameter Grids

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

# Load the Dataset

Load the dataset from a CSV file and convert the 'DateTime' column to a datetime object.

```python
data = pd.read_csv('model_data.csv')
data['DateTime'] = pd.to_datetime(data['DateTime'])
```

# Classification

Perform classification using grouped time series cross-validation with the defined pipelines and parameter grids.

```python
from grouped_timeserie_cv import GroupedTimeSerieCV
grouped_cv = GroupedTimeSerieCV()
result = grouped_cv.classify(data, pipelines, param_grids, 'D', 'DateTime','Label', 'accuracy')
```
