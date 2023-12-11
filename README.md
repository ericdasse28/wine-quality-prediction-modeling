# Wine Quality Prediction Modeling

Implementation of a model that predicts the quality of a red wine on a scale of 0-10,
given a set of physicochemical features of wine as inputs

## Dataset

The dataset used is the [Wine Quality dataset](https://archive.ics.uci.edu/dataset/186/wine+quality),
provided the UCI Machine Learning repository.

## Data and model versioning

We use [DVC](https://realpython.com/python-data-version-control/) to version data and the ML model.
It is listed among the Python dependencies of this project

The remote repository used for data and model
is [DagsHub](https://dagshub.com/ericdasse28/wine-quality-prediction-modeling/src/main)

To put a file or a directory under DVC tracking, use the command line:

```bash
dvc add <file or directory>
```

To push DVC-tracked files to the remote repository, use the command line:

```bash
dvc push -r origin
```

You can also directly set _origin_ as the default DVC remote using the command:

```bash
dvc remote default origin
```

Hence, pushing becomes simpler: `dvc push`
