from sklearn.ensemble import RandomForestClassifier
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error, roc_auc_score
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

key = "water_potability"
run = Run.get_context()
ws = run.experiment.workspace

# +
from sklearn.impute import SimpleImputer

if key in ws.datasets.keys(): 
        print("Found dataset!")
        dataset = ws.datasets[key] 
else:
    exit(1)
    
x_raw = dataset.to_pandas_dataframe()
x_imputed = SimpleImputer(strategy='mean').fit_transform(x_raw)
x = pd.DataFrame(x_imputed, columns=x_raw.columns)
y = x.pop("Potability")


# -

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    #parser.add_argument('--data-folder', type=str, dest='data_folder', help='data folder mount point')
    parser.add_argument('--n_estimators', type=int, default=10, help="Number of trees in the forest")
    parser.add_argument('--max_depth', type=int, default=5, help="The maximum depth of the tree")
    
    args = parser.parse_args()

    run.log("Number of estimators:", np.int(args.n_estimators))
    run.log("Max depth:", np.int(args.max_depth))
    
    #x, y = get_data(args.data_folder)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

    model = RandomForestClassifier(n_estimators=args.n_estimators, 
                                   max_depth=args.max_depth,
                                   random_state=4324).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    y_pred = model.predict(x_test)
    auc_weighted = roc_auc_score(y_test, y_pred, average="weighted")
    print("Params: ")
    print(args)
    print("Predictions:")
    print(model.predict(x_test))
    print("accuracy: "+str(accuracy))
    run.log("AUC_weighted", np.float(auc_weighted))

    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model, 'outputs/model_{}_{}.joblib'.format(args.n_estimators, args.max_depth))

if __name__ == '__main__':
    main()
