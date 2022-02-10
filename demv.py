import numpy as np
import pandas as pd


def balance_set(w_exp, w_obs, df: pd.DataFrame, tot_df, round_level=None, debug=False, k=-1):
    disp = round(w_exp / w_obs, round_level) if round_level else w_exp / w_obs
    disparity = [disp]
    i = 0
    while disp != 1 and i != k:
        if w_exp / w_obs > 1:
            df = df.append(df.sample())
        elif w_exp / w_obs < 1:
            df = df.drop(df.sample().index, axis=0)
        w_obs = len(df) / len(tot_df)
        disp = round(
            w_exp / w_obs, round_level) if round_level else w_exp / w_obs
        disparity.append(disp)
        if debug:
            print(w_exp / w_obs)
        i += 1
    return df, disparity, i


def sample(d: pd.DataFrame, s_vars: list, label: str, round_level: float, debug: bool = False, i: int = 0, G: list = [], cond: bool = True, stop=-1):
    d = d.copy()
    n = len(s_vars)
    disparities = []
    iter = 0
    if i == n:
        for l in np.unique(d[label]):
            g = d[(cond) & (d[label] == l)]
            w_exp = (len(d[cond])/len(d)) * (len(d[d[label] == l])/len(d))
            w_obs = len(g)/len(d)
            g_new, disp, k = balance_set(
                w_exp, w_obs, g, d, round_level, debug, stop)
            disparities.append(disp)
            G.append(g_new)
            iter = max(iter, k)
        return G, iter
    else:
        s = s_vars[i]
        i = i+1
        G1, k1 = sample(d, s_vars, label, round_level, debug, i,
                        G.copy(), cond=cond & (d[s] == 0), stop=stop)
        G2, k2 = sample(d, s_vars, label, round_level, debug, i,
                        G.copy(), cond=cond & (d[s] == 1), stop=stop)
        G += G1
        G += G2
        iter = max([iter, k1, k2])
        limit = 1
        for s in s_vars:
            limit *= len(np.unique(d[s]))
        if len(G) == limit*len(np.unique(d[label])):
            return pd.DataFrame(G.pop().append([g for g in G]).sample(frac=1, random_state=2)), disparities, iter
        else:
            return G, iter


class DEMV:
    def __init__(self, round_level=None, debug=False, stop=-1):
        self.round_level = round_level
        self.debug = debug
        self.disparities = None
        self.stop = stop
        self.iter = 0

    def predict(self, dataset):
        return dataset

    def transform(self, dataset):
        return dataset

    def fit_transform(self, dataset: pd.DataFrame, protected_attrs: list, label_name: str):
        df_new, disparities, iter = sample(dataset, protected_attrs,
                                           label_name, self.round_level, self.debug, 0, [], True, self.stop)
        self.disparities = disparities
        self.iter = iter
        return df_new

    def get_disparities(self):
        return self.disparities

    def get_iters(self):
        return self.iter
