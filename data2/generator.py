import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
import argparse

# BUILD A SYNTHETIC DATASET

parser = argparse.ArgumentParser(description="Generate a synthetic dataset")
parser.add_argument('-s', '--samples', type=int, help='Number of samples')
parser.add_argument('-c', '--classes', type=int, help='Number of classes')
parser.add_argument('-f', '--features', type=int, help='Number of features')
parser.add_argument('-i', '--informatives', type=int,
                    help='Number of informative features')
parser.add_argument('-sv', '--sensitive_vars', type=int,
                    help='Number of sensitive variables')
parser.add_argument('-n', '--name', type=str, help='Name of the file')
args = parser.parse_args()

data = make_classification(n_samples=args.samples, n_features=args.features,
                           n_classes=args.classes, n_informative=args.informatives)
df = pd.DataFrame(data[0])
i = 0
for vars in range(args.sensitive_vars):
    i += 1
    sens_var0 = np.full(shape=round(args.samples/2), fill_value=0, dtype=int)
    sens_var1 = np.full(shape=round(args.samples/2), fill_value=1, dtype=int)
    df['s'+str(i)] = np.hstack((sens_var0, sens_var1))
    df = df.sample(frac=1)
df['y'] = data[1]
df.to_csv(args.name+'.csv', index=False)
