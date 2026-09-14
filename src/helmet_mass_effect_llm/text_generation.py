import os
import sys
import pandas as pd
import numpy as np
import torch
import datasets
from datasets import Dataset
from pydantic import confloat, validate_arguments
from typing import Literal
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import tensor as tnsr
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchmetrics.classification import MulticlassAUROC, MulticlassAccuracy, MulticlassAveragePrecision
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification, BioGptForSequenceClassification
from transformers import TrainingArguments, Trainer, get_scheduler, DataCollatorForLanguageModeling
from tqdm.auto import tqdm
import evaluate
import wandb


from helmet_mass_effect_pred.evaluation.cv_results import (
    get_results_per_cv,
    target_transform_partial,
)
from helmet_mass_effect_pred.data.splitting import drop_missing_impute_and_split
from helmet_mass_effect_pred.data.external_cohort import create_MGH_and_BMC
from helmet_mass_effect_pred.evaluation.scoring import OTV_scorer
from helmet_mass_effect_pred.evaluation.visualisation import make_tsne_and_pca

time_NaN_fill = -10

@validate_arguments
def create_OLTVs(
    filter_out_less_than_24_hours = True, 
    disable_one_hot_encoding = False,
    time_NaN_fill: int= -10,
    use_only_rc_values = True,
    lookback_hours = 18,
    lookahead_hours = 24,
    use_lsw_as_base_time_instead_of_pres = True,
    create_new_mls_and_pgs_vals = True,
    use_ffil_and_rolling = True,
    dont_transform_otv = False,
    add_min_values = False,
    use_mode_as_feature: bool = False,
    use_prev_as_feature: bool = False,
    preserve_true_scans: bool = False
):
    dt_parser = lambda x: pd.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")

    LTV = pd.read_json("data/processed/mgh_ltv_2.0_text.json")
    bmc_LTV = pd.read_csv("data/raw/LTV_BMC_2.csv")
    OTV = pd.read_csv("data/raw/OTV_2.1.csv")
    bmc_OTV = pd.read_csv("data/raw/OTV_BMC_3.csv")

    LTV = LTV.dropna(how = "all", axis = 1)
    bmc_LTV = bmc_LTV.dropna(how = "all", axis = 1)

    bmc_LTV = bmc_LTV.replace("NA", np.nan)
    bmc_OTV = bmc_OTV.replace("NA", np.nan)

    bmc_LTV["surg"] = np.nan

    events = ["surgdt", "encounter_id", "CDW_ID", "CDW_MRN", "LSW", "first_arrival_dt", "DISCH_DT", "startdt", "redcap_id"]

    surg_events = bmc_OTV[bmc_OTV["surg"] == 1][events].rename(columns = {"surgdt": "dt"})
    surg_events["surg"] = 1

    bmc_LTV = pd.concat([bmc_LTV, surg_events], ignore_index = True)
    bmc_LTV = bmc_LTV.sort_values(["encounter_id", "dt"]).reset_index(drop = True)

    bmc_LTV["hts23"] = time_NaN_fill
    bmc_LTV.rename(columns={"glucose": "gluc", "sodium": "na"}, inplace=True)
    bmc_LTV.rename(columns={"creatinine": "cr", "osmolality": "osm"}, inplace=True)
    bmc_LTV.rename(columns={"CDW_MRN": "ptid", "LSW": "lsw"}, inplace=True)
    
    if not use_only_rc_values:
        bmc_LTV.rename(columns={"size_max_pgs": "size_pgs"}, inplace=True)

    bmc_OTV.rename(columns={"CDW_MRN": "ptid", "LSW": "lsw"}, inplace=True)
    bmc_OTV.rename(columns={"CMO": "CMO_disch", "CMO_DT": "CMO_dt"}, inplace=True)
    bmc_OTV.rename(columns={"DNR": "DNR_disch", "DNR_DT": "DNR_dt"}, inplace=True)
    bmc_OTV.rename(columns={"DISCH_DT": "Discharge_Date"}, inplace=True)
    bmc_OTV.rename(columns={"X.24.hours": "<24 hours"}, inplace=True)
    bmc_OTV.rename(columns={"first_arrival_dt": "pres"}, inplace=True)

    nonzero_mls = bmc_LTV[bmc_LTV["size_mls"] > 0]
    nonzero_mls = nonzero_mls.groupby("ptid").first().reset_index()
    nonzero_mls = nonzero_mls[["ptid", "size_mls", "dt"]]
    nonzero_mls.rename(columns={"size_mls": "firstmls"}, inplace=True)
    nonzero_mls.rename(columns={"dt": "firstmlsdt"}, inplace=True)

    bmc_OTV = pd.merge(bmc_OTV, nonzero_mls, on="ptid", how="left")

    bmc_LTV["lsw"] = pd.to_datetime(bmc_LTV["lsw"])
    bmc_LTV["dt"] = pd.to_datetime(bmc_LTV["dt"])
    bmc_LTV["obsTime"] = bmc_LTV["dt"] - bmc_LTV["lsw"]
    bmc_LTV["obsTime"] = bmc_LTV["obsTime"].dt.total_seconds() // 3600

    def get_rc_columns(df):
        rc_columns = [column for column in df.columns if "rc_" in column]
        non_rc_versions = [column.replace("rc_", "") for column in rc_columns]
        non_rc_versions = [col for col in non_rc_versions if col in df.columns]
        df = df.drop(non_rc_versions, axis=1)
        df.columns = [column.replace("rc_", "") for column in df.columns]
        return df

    def mls_pgs_convert(df):
        if "mls" in df.columns:
            df["size_mls"] = df[["mls", "size_mls"]].max(axis=1).copy()
            df = df.drop("mls", axis=1)
        if "pgs" in df.columns:
            df["size_pgs"] = df[["pgs", "size_pgs"]].max(axis=1).copy()
            df = df.drop("pgs", axis=1)
        return df
    
    if use_only_rc_values:
        LTV = get_rc_columns(LTV)
        bmc_LTV = get_rc_columns(bmc_LTV)
        OTV = get_rc_columns(OTV)
        bmc_OTV = get_rc_columns(bmc_OTV)

    LTV = mls_pgs_convert(LTV)
    bmc_LTV = mls_pgs_convert(bmc_LTV)

    LTV.iloc[:, 4:-1] = LTV.iloc[:, 4:-1].fillna(time_NaN_fill)
    bmc_LTV = bmc_LTV.fillna(time_NaN_fill)

    OLTV = pd.merge(LTV, OTV, on="ptid", how="inner")
    bmc_OLTV = pd.merge(bmc_LTV, bmc_OTV, on="ptid", how="inner")
    OLTV.rename(columns={"first_osmotic": "first_osmotic_dt"}, inplace=True)

    del LTV, bmc_LTV

    columns = list(set(OLTV.columns).intersection(set(bmc_OLTV.columns)))
    bmc_columns = columns.copy()
    columns.append("rad_text")
    mgh_columns = columns
    OLTV = OLTV[mgh_columns]
    bmc_OLTV = bmc_OLTV[bmc_columns]

    base_time, secondary_time = (
        ("lsw", "pres") if use_lsw_as_base_time_instead_of_pres else ("pres", "lsw")
    )

    mgh_times = [secondary_time] + [col for col in mgh_columns if col.endswith("dt")]
    bmc_times = [secondary_time] + [col for col in bmc_columns if col.endswith("dt")]

    def generate_OLTV(DF, preserved_otv_columns):
        DF.rename(columns = {"lsw_x": "lsw"}, inplace = True)
        DF.drop("lsw_y", axis = 1, inplace = True)
        DF.drop("surg_x", axis = 1, inplace = True)
        preserved_otv_columns.append("surg_y")
        preserved_otv_columns.append("lsw")

        outcome_cols = ["parenchymal_dt", "petechial_dt", "maxmlsdt"]
        outcome_cols += ["homeDischarge", "hemorrhage", "bleed"]
        outcome_cols += ["lvo_location", "CMO_disch", "deathByDischarge", "lastmls"]
        outcome_cols += ["longTermCareByDischarge", "lastmlsdt", "ph2_dt", "maxmls"]
        outcome_cols += ["rehabByDischarge", "deathOrHospice", "hospiceByDischarge"]
        outcome_cols += ["DNR_dt", "CMO_dt", "DNR_disch", "Discharge_Date", "tt_mt"]

        DF = DF.drop(outcome_cols, axis=1)
        DF = DF.reindex(sorted(DF.columns), axis=1)

        columns = DF.columns
        times = [secondary_time] + [col for col in columns if col.endswith("dt")]

        DF = DF[DF["<24 hours"] == 1] if filter_out_less_than_24_hours else DF

        DF[DF.ethnicity == "Hispanic"]["Race"] = "Hispanic"
        DF = DF.drop(["ethnicity", "<24 hours"], axis = 1)
        DF = DF.dropna(how="all", axis=1)
        DF = DF.drop([column for column in DF.columns if "ph2" in column], axis=1)
        DF = DF.drop([column for column in DF.columns if "ph1" in column], axis=1)
        columns = [column for column in DF.columns if "tt_" not in column]

        DF[base_time] = pd.to_datetime(DF[base_time])
        for time in times:
            DF[time] = pd.to_datetime(DF[time])
            DF[time] = DF[time] - DF[base_time]
            DF[time] = DF[time].dt.total_seconds() // 3600
            DF[time] = DF[time].fillna(int(time_NaN_fill))
        DF = DF.drop(base_time, axis=1)
        DF["dt"] = DF["dt"].astype(int)
        columns = DF.columns
        DF = DF[columns]

        def process_values(DF, prefix, values, size_column, time_NaN_fill):
            for val in values:
                col = prefix + str(val)
                coldt = col + "dt"
                DF[col] = 0
                DF[coldt] = 1e10
                DF[col] = DF.apply(
                    lambda x: 1 if x[size_column] >= val else x[col], axis=1
                )
                DF[col] = DF.groupby("ptid")[col].transform("max")
                DF[coldt] = DF.apply(
                    lambda x: x["dt"] if x[size_column] >= val else x[coldt], axis=1
                )
                DF[coldt] = DF.groupby("ptid")[coldt].transform("min")
                DF[coldt] = DF[coldt].replace(1e10, time_NaN_fill)
            return DF

        if create_new_mls_and_pgs_vals:
            DF = process_values(DF, "mls", [3, 9, 12, 15], "size_mls", time_NaN_fill)
            DF = process_values(DF, "pgs", [2, 4, 6, 8, 10], "size_pgs", time_NaN_fill)

            mls_cols = [f"mls{i}" for i in [3, 9, 12, 15]]
            mls_dt_cols = [f"mls{i}dt" for i in [3, 9, 12, 15]]
            pgs_cols = [f"pgs{i}" for i in [2, 4, 6, 8, 10]]
            pgs_dt_cols = [f"pgs{i}dt" for i in [2, 4, 6, 8, 10]]

            preserved_otv_columns += mls_cols + pgs_cols + mls_dt_cols + pgs_dt_cols

        if disable_one_hot_encoding:
            for column in DF.columns:
                if DF[column].dtype == "object":
                    print("dropping ", column)
                    DF = DF.drop(column, axis=1)
                    preserved_otv_columns.remove(column)
        else:
            for column in columns:
                if DF[column].nunique() <= 10:
                    if DF[column].nunique() > 3:
                        print("one hot encoding ", column)
                        DF = pd.get_dummies(DF, columns=[column], drop_first=True)
        for column in DF.columns:
            if DF[column].dtype == "bool":
                DF[column] = DF[column].astype(int)

        time_columns = [column for column in DF.columns if column.endswith("dt")]
        time_columns = [column for column in time_columns if column != "dt"]
        time_column_indices = [DF.columns.get_loc(column) for column in time_columns]
        maybe_val_indices = [i - 1 for i in time_column_indices]

        for index, name in enumerate(time_columns):
            # remove dt or potential _ from column name
            print("hiding future times for ", name)
            time_string = name.replace("_dt", "")
            time_string = time_string.replace("dt", "")

            maybe_val_index = maybe_val_indices[index]
            if time_string in DF.columns[maybe_val_index]:
                val_col_name = DF.columns[maybe_val_index]
                print("found ", val_col_name)
                DF[val_col_name].loc[DF["dt"] <= DF[name]] = int(time_NaN_fill)
                DF[val_col_name].loc[DF[val_col_name] == 0] = int(time_NaN_fill)
                DF.rename(
                    columns={val_col_name: val_col_name + "_time_censored"},
                    inplace=True,
                )
                preserved_otv_columns.append(val_col_name + "_time_censored")
                preserved_otv_columns.remove(val_col_name)

            DF[name].loc[DF["dt"] <= DF[name]] = int(time_NaN_fill)
            DF.rename(columns={name: name + "_time_censored"}, inplace=True)
            preserved_otv_columns.append(name + "_time_censored")
            preserved_otv_columns.remove(name)
        
        # group by ptid and hour
        try:
            DF = DF.groupby(["ptid", "dt"]).max().reset_index()
        except:
            DF = DF.groupby(["ptid", "dt"]).first().reset_index()
        DF = DF.set_index(["ptid", "dt"])

        # reindex missing hours
        full_index = pd.MultiIndex(
            levels=[[], []], codes=[[], []], names=["ptid", "dt"]
        )
        for ptid in DF.index.get_level_values("ptid").unique():
            min_dt = DF.loc[ptid].index.min()
            max_dt = DF.loc[ptid].index.max()
            idx = pd.MultiIndex.from_product(
                [[ptid], range(min_dt, max_dt + 1)], names=["ptid", "dt"]
            )
            full_index = full_index.append(idx)

        DF = DF.reindex(full_index).reset_index()

        target = (
            DF.groupby(["ptid"])["size_mls"]
            .shift(-lookahead_hours)
            .rolling(lookahead_hours, min_periods=1)
            .max()
        )

        if use_ffil_and_rolling:
            otv_columns = [
                column for column in DF.columns if column in preserved_otv_columns
            ]
            otv_columns.remove("ptid")

            if dont_transform_otv:
                df_otv = DF[otv_columns].copy()
                source_DF = DF.drop(otv_columns, axis=1).copy()
            else:
                source_DF = DF.copy()

            DF_rolling = source_DF.groupby("ptid").apply(
                lambda x: x.rolling(lookback_hours, min_periods=1).max()
            )
            if add_min_values:
                min_df = source_DF.copy()
                # replace all time_NaN_fill with NaN
                min_df = min_df.replace(int(time_NaN_fill), np.nan)
                DF_min = min_df.groupby("ptid").apply(
                    lambda x: x.rolling(lookback_hours, min_periods=1).min()
                )

            DF_ffill = (
                source_DF.replace(int(time_NaN_fill), np.nan).groupby("ptid").ffill()
            )
            DF_ffill = DF_ffill.fillna(int(time_NaN_fill))

            DF_ffill = DF_ffill.drop(["dt"], axis=1)
            DF_ffill.columns = ["ffill_" + column for column in DF_ffill.columns]

            # add rolling to name of every col in DF_rolling
            DF_rolling = DF_rolling.drop(["obsTime"], axis=1)
            DF_rolling.columns = ["rolling_" + column for column in DF_rolling.columns]
            DF_rolling.rename(columns={"rolling_ptid": "ptid"}, inplace=True)
            DF_rolling.rename(columns={"rolling_dt": "dt"}, inplace=True)

            if add_min_values:
                DF_min = DF_min.drop(["obsTime"], axis=1)
                DF_min.columns = ["min_" + column for column in DF_min.columns]
                DF_min = DF_min.drop(["min_ptid", "min_dt"], axis=1)
                DF_rolling = pd.concat([DF_rolling, DF_min], axis=1)

            del DF
            del source_DF

            DF_rolling = DF_rolling.drop('ptid', axis=1).reset_index()
            DF_rolling = DF_rolling.drop('level_1', axis=1)         

            if dont_transform_otv:

                DF = pd.concat(
                    [DF_rolling, DF_ffill, df_otv],
                    axis=1,
                )
            else:
                DF = pd.concat([DF_rolling, DF_ffill], axis=1)

        DF["target"] = target
        DF = DF.dropna(subset=["target"])
        DF = DF[DF.target != time_NaN_fill]

        DF = DF.dropna(
            how="all",
            subset=[column for column in DF.columns if column != "target"],
        )

        # drop rows that are duplicates in all values except dt and obsTime
        DF = DF.drop_duplicates(
            subset=[column for column in DF.columns if column not in ["dt", "obsTime"]],
            keep="last",
        )

        DF = DF.reset_index(drop=True)
        target = DF["target"].copy()
        DF = DF.drop("target", axis=1)
        return DF, target
    
    preserved_bmc_otv_columns = list(
        set(OTV.columns).intersection(set(bmc_OLTV.columns))
    )

    presevered_otv_columns = list(set(OTV.columns).intersection(set(OLTV.columns)))

    OLTV, target = generate_OLTV(OLTV, presevered_otv_columns)
    bmc_OLTV, bmc_target = generate_OLTV(bmc_OLTV, preserved_bmc_otv_columns)

    def process_dataframe(df, times, base_time, time_NaN_fill):
        df[base_time] = pd.to_datetime(df[base_time])
        for time in times:
            if time in list(df.columns):
                df[time] = pd.to_datetime(df[time])
                df[time] = (df[time] - df[base_time]).dt.total_seconds() // 3600
                df[time] = df[time].fillna(int(time_NaN_fill))
        return df
    
    OTV = process_dataframe(OTV, times, base_time, time_NaN_fill)
    bmc_OTV = process_dataframe(bmc_OTV, bmc_times, base_time, time_NaN_fill)

    if use_mode_as_feature:
        OLTV['target_mode'] = target.mode().values.item()
        bmc_OLTV['target_mode'] = bmc_target.mode().values.item()
    
    if use_prev_as_feature:
        prev_values = np.zeros(shape = [OLTV.shape[0], ])
        prev_pt = None
        for i, pt in enumerate(OLTV['ptid']):
            if prev_pt is None or prev_pt != pt:
                prev_values[i] = 0
            else:
                prev_values[i] = OLTV.iloc[i]['ffill_size_mls']
            prev_pt = pt
        
        prev_values_bmc = np.zeros(shape = [bmc_OLTV.shape[0], ])
        prev_pt = None
        for i, pt in enumerate(bmc_OLTV['ptid']):
            if prev_pt is None or prev_pt != pt:
                prev_values_bmc[i] = 0
            else:
                prev_values_bmc[i] = bmc_OLTV.iloc[i]['ffill_size_mls']
            prev_pt = pt

        OLTV['prev_mls'] = prev_values
        bmc_OLTV['prev_mls'] = prev_values_bmc

    return OLTV, target, bmc_OLTV, bmc_target, OTV, bmc_OTV


def get_cleaned_data(
    project: str = "bmc_data_exps",
    OTV_version: Literal["1.1", "2.0", "2.1"] = "2.1",
    LTV_version: Literal["1.1", "2.0"] = "2.0",
    disable_one_hot_encoding: bool = True, 
    impute_method: Literal["simple", "iterative", "knn"] = "simple",
    time_NaN_fill: int = -10,
    use_only_rc_values = True,
    lookback_hours = 24,
    lookahead_hours = 24,
    col_na_drop_fraction: confloat(ge=0.0, le = 1.0) = 0.3,
    patient_na_drop_fraction: confloat(ge=0.0, le=1.0) = 0.2,
    data_standard_scaling: bool = False,
    use_lsw_as_base_time_instead_of_pres: bool = True,
    create_new_mls_and_pgs_vals: bool = True,
    create_pca_and_tsne: bool = False,
    make_shap_plots: bool = False,
    use_ffil_and_rolling: bool = True,
    remove_data_after_surgery_occurs: bool = True,
    add_min_values: int = 0,
    use_mode_as_feature: bool = True,
    use_prev_as_feature: bool = True,
    use_avg_kernel_as_feature: bool = True,
    use_regression_as_feature: bool = True,
    remove_excess_features: bool = True,
):

    target_transform = lambda x: target_transform_partial(x, time_NaN_fill)

    if add_min_values > 0:
        add_min_values = True
    else:
        add_min_values = False
    
    OLTV, targets, bmc_OLTV, bmc_targets, OTV, bmc_OTV = create_MGH_and_BMC(
        filter_out_less_than_24_hours=True,
        disable_one_hot_encoding=disable_one_hot_encoding,
        time_NaN_fill=time_NaN_fill,
        use_only_rc_values=use_only_rc_values,
        lookback_hours=lookback_hours,
        lookahead_hours=lookahead_hours,
        use_lsw_as_base_time_instead_of_pres=use_lsw_as_base_time_instead_of_pres,
        create_new_mls_and_pgs_vals=create_new_mls_and_pgs_vals,
        use_ffil_and_rolling=use_ffil_and_rolling,
        add_min_values=add_min_values,
        use_mode_as_feature=use_mode_as_feature,
        use_prev_as_feature=use_prev_as_feature,
    )

    if add_min_values:
        OLTV = OLTV[bmc_OLTV.columns]

    base_time = "LSW" if use_lsw_as_base_time_instead_of_pres else "first_arrival_dt"

    if use_ffil_and_rolling:
        # remove ffill_ from column names
        OLTV.columns = [x.replace("ffill_", "") for x in OLTV.columns]
        bmc_OLTV.columns = [x.replace("ffill_", "") for x in bmc_OLTV.columns]

    OLTV_columns = OLTV.columns
    bmc_OLTV_columns = bmc_OLTV.columns

    targets = targets.apply(target_transform)
    bmc_targets = bmc_targets.apply(target_transform)

    test_frac = 0.2 

    def split_and_impute(OLTV, targets, test_frac):
        OLTV = pd.concat([OLTV, targets], axis=1)
        num_rows, num_cols = OLTV.shape

        OLTV = OLTV.reset_index(drop=True)
        unique = OLTV.ptid.unique()
        ptid_train, ptid_test = train_test_split(unique, test_size=test_frac, shuffle=False)
        train_test_list = [[ptid_train, ptid_test]]

        train_OLTV = OLTV[OLTV.ptid.isin(ptid_train)].copy()
        test_OLTV = OLTV[OLTV.ptid.isin(ptid_test)].copy()

        train_targets = train_OLTV['target'].copy()
        test_targets = test_OLTV['target'].copy()

        train_OLTV = train_OLTV.drop(columns=['target'], axis=1)
        test_OLTV = test_OLTV.drop(columns=['target'], axis=1)

        train_ptid = train_OLTV["ptid"].copy()
        test_ptid = test_OLTV["ptid"].copy()

        train_OLTV = train_OLTV.drop(columns=["ptid"], axis=1)
        test_OLTV = test_OLTV.drop(columns=["ptid"], axis=1)

        cols = train_OLTV.columns

        imputer = SimpleImputer().fit(train_OLTV)

        train_OLTV = imputer.transform(train_OLTV)
        test_OLTV = imputer.transform(test_OLTV)

        train_OLTV = pd.DataFrame(train_OLTV, columns = cols)
        test_OLTV = pd.DataFrame(test_OLTV, columns = cols)

        return train_ptid, test_ptid, train_OLTV, test_OLTV, train_targets, test_targets

    train_ptid, test_ptid, train_OLTV, test_OLTV, train_targets, test_targets = split_and_impute(OLTV=OLTV, targets=targets, test_frac = test_frac)
    bmc_train_ptid, bmc_test_ptid, bmc_train_OLTV, bmc_test_OLTV, bmc_train_targets, bmc_test_targets = split_and_impute(OLTV=bmc_OLTV, targets = bmc_targets, test_frac=test_frac)
    
    bmc_pitd = pd.concat([bmc_train_ptid, bmc_test_ptid])
    bmc_OLTV = pd.concat([bmc_train_OLTV, bmc_test_OLTV])
    bmc_targets = pd.concat([bmc_train_targets, bmc_test_targets])

    train_OLTV = pd.DataFrame(train_OLTV, columns = OLTV_columns)
    test_OLTV = pd.DataFrame(test_OLTV, columns = OLTV_columns)

    return train_OLTV, train_targets, test_OLTV, test_targets, bmc_OLTV, bmc_targets

def generate_labels(OLTV, targets):
    target_transform = lambda x: target_transform_partial(x, time_NaN_fill)
    transform_mls = OLTV['size_mls'].apply(target_transform)
    targets = targets.reset_index(drop = True)
    OLTV = OLTV.reset_index(drop = True)

    labels = []
    for i, val in enumerate(transform_mls):
        if val < targets[i]:
            labels.append("worsen")
        elif val == targets[i]:
            labels.append("stay the same")
        elif val > targets[i]:
            labels.append("improve")
    
    return labels

def data_to_text(df, lookback, lookahead):
    text_array = []
    df = df.reset_index(drop = True)
    for row in range(df.shape[0]):
        dt = df['dt'][row]
        age = df['age_calc'][row]
        lbs = df['lbs'][row]
        sex = df['sex'][row]
        af = df['af'][row]

        mls = df['size_mls'][row]
        prev_mls = df['prev_mls'][row]
        pgs = df['size_pgs'][row]


        stroke_territory = df['stroke_territory'][row]

        if stroke_territory == 0:
            stroke_territory = "minimal"
        elif stroke_territory == 1:
            stroke_territory = "low"
        elif stroke_territory == 2:
            stroke_territory = "moderate"
        elif stroke_territory == 3:
            stroke_territory = "severe"
        else:
            stroke_territory = "indeterminate"
        
        #NIH stoke scale score (on admission)
        nihss = df['nihss'][row]

        #Alberta stroke program early CT score
        aspects = df['aspects'][row]

        def make_delta(now, roll):
            if now < roll:
                delt = "decreased"
            elif now == roll:
                delt = "had no change"
            elif now > roll:
                delt = "increased"
            
            return delt
        
        #mean arterial pressure
        map1 = df['map1'][row]
        map = df['map'][row]
        roll_map = df['rolling_map'][row]

        delta_map = make_delta(map, roll_map)

        #systolic 
        sbp1 = df['sbp1'][row]
        sbp = df['sbp'][row]
        roll_sbp = df['rolling_sbp'][row]

        delta_sbp = make_delta(sbp, roll_sbp)

        #diastolic
        dbp1 = df['dbp1'][row]
        dbp = df['dbp'][row]
        roll_dbp = df['rolling_dbp'][row]

        delta_dbp = make_delta(dbp, roll_dbp)

        #heart rate
        pulse1 = df['pulse1'][row]
        pulse = df['pulse'][row]
        roll_pulse = df['rolling_pulse'][row]

        delta_pulse = make_delta(pulse, roll_pulse)

        #temperature
        temp1 = df['temp1'][row]
        temp = df['temp'][row]
        roll_temp = df['rolling_temp'][row]

        delta_temp = make_delta(temp, roll_temp)
        
        #serum osmolality
        osm1 = df['osm1'][row]
        osm = df['osm'][row]
        roll_osm = df['rolling_osm'][row]

        delta_osm = make_delta(osm, roll_osm)

        #sodium
        na1 = df['na1'][row]
        na = df['na'][row]
        roll_na = df['rolling_na'][row]

        delta_na = make_delta(na, roll_na)

        #blood urea nitrogen
        bun1 = df['bun1'][row]
        bun = df['bun'][row]
        roll_bun = df['rolling_bun'][row]

        delta_bun = make_delta(bun, roll_bun)

        #creatinine
        cr1 = df['cr1'][row]
        cr = df['cr'][row]
        roll_cr = df['rolling_cr'][row]

        delta_cr = make_delta(cr, roll_cr)

        #blood glucose
        gluc1 = df['gluc1'][row]
        gluc = df['gluc'][row]
        roll_gluc = df['rolling_gluc'][row]

        delta_gluc = make_delta(gluc, roll_gluc)

        #white blood cell count
        wbc1 = df['wbc1'][row]
        wbc = df['wbc'][row]
        roll_wbc = df['rolling_wbc'][row]

        delta_wbc = make_delta(wbc, roll_wbc)

        if sex == 0:
            sex = "male"
        elif sex == 1:
            sex = "female"
        else:
            sex = ""
        af = "with a history of atrial fibrillation" if af else ""


        text = "".join([f"The patient is a {age} year old {sex} {af} who weighs {lbs} pounds. ",
                f"They were admitted to the hospital {dt} hours ago after an suffering a {stroke_territory} ischemic stroke. ",
                f"At the time of admission, the patient had a Alberta stroke program early CT scan score of {aspects} and a NIH stroke score of {nihss}. ",
                f"In the last {lookback} hours, the patient's mean arterial blood pressure has {delta_map}, ",
                f"systolic blood pressure has {delta_sbp}, ",
                f"diastolic blood pressure has {delta_dbp}, ",
                f"heart rate has {delta_pulse}, ",
                f"and body temperature has {delta_temp}. ",
                f"The patient's current mean arterial blood pressure is {map}, ",
                f"systolic blood pressure is {sbp}, ",
                f"diastolic blood pressure is {dbp}, ",
                f"heart rate is {pulse}, ",
                f"and body temperature is {temp}. ",
                f"In the last {lookback} hours, the patient's serum osmolality has {delta_osm}, ",
                f"sodium has {delta_na}, ",
                f"blood urea nitrogen has {delta_bun}, ",
                f"blood glucose has {delta_gluc}, ",
                f"creatinine has {delta_cr}, ",
                f"and white blood cell count has {delta_wbc}. ",
                f"The patient's current serum osmolality is {osm}, ",
                f"sodium is {na}, ",
                f"blood urea nitrogen is {bun}, ",
                f"blood glucose is {gluc}, ",
                f"creatinine is {cr}, ",
                f"and white blood cell count is {wbc}. ",
                f"The patient's current midline shift of the septum pellucidum is {mls} mm, and the previous measured midline shift value was {prev_mls} mm. ",
                f"The patient's pineal gland shift is currently {pgs} mm. ",
                f"Based on the patient data, we expect that over the next {lookahead} hours, the midline shift will <mask>."
        ])
        
        text = text.replace("-10.0", "unknown")
        text_array.append(text)

    return text_array

def make_dataset(text, labels, split):
    data = pd.DataFrame({'text': text, 'labels': labels})
    return Dataset.from_pandas(data, split = split)

def get_sets_from_OLTV(train_OLTV, train_targets, test_OLTV, test_targets, bmc_OLTV, bmc_targets):
    train_text = data_to_text(train_OLTV, 18, 24)
    test_text = data_to_text(test_OLTV, 18, 24)
    bmc_text = data_to_text(bmc_OLTV, 18, 24)

    train_labels = generate_labels(train_OLTV, train_targets)
    test_labels = generate_labels(test_OLTV, test_targets)
    bmc_labels = generate_labels(bmc_OLTV, bmc_targets)

    train_set = make_dataset(train_text, train_labels, 'train')
    test_set = make_dataset(test_text, test_labels, 'test')
    bmc_set = make_dataset(bmc_text, bmc_labels, 'eval')

    return train_set, test_set, bmc_set


def fine_tune_cl(train_set, test_set, bmc_set):
    cl_tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-Longformer")
    cl_model = AutoModelForMaskedLM.from_pretrained("yikuan8/Clinical-Longformer", num_labels= 4, problem_type="multi_label_classification")

    def cl_tokenize_function(examples):
        return cl_tokenizer(examples(['text', 'labels']), padding="max_length", max_length = 750, truncation=True)
    
    train_set = train_set.map(cl_tokenize_function, batched = True)
    train_set = train_set.remove_columns(['text']).select(range(1000))
    train_set.set_format("torch")

    test_set = test_set.map(cl_tokenize_function, batched = True)
    test_set = test_set.remove_columns(['text']).select(range(500))
    test_set.set_format("torch")
    
    bmc_set = bmc_set.map(cl_tokenize_function, batched = True)
    bmc_set = bmc_set.remove_columns(['text'])
    bmc_set.set_format("torch")

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    
    training_args = TrainingArguments(output_dir="models/fine_tuning_cl", eval_strategy="epoch", run_name="run0_cl")

    trainer = Trainer(
        model=cl_model,
        args=training_args,
        train_dataset= train_set,
        eval_dataset= test_set,
        compute_metrics=compute_metrics,
    )

    results = trainer.train()


def fine_tune_bgpt(train_set, test_set, bmc_set):

    bgpt_tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
    bgpt_model = BioGptForSequenceClassification.from_pretrained("microsoft/biogpt", num_labels= 4, problem_type="multi_label_classification")

    def bgpt_tokenize_function(examples):
        return bgpt_tokenizer(examples["text"], padding="max_length", max_length = 750, truncation=True)

    train_set = train_set.map(bgpt_tokenize_function, batched = True)
    train_set = train_set.remove_columns(['text']).select(range(1000))
    train_set.set_format("torch")

    test_set = test_set.map(bgpt_tokenize_function, batched = True)
    test_set = test_set.remove_columns(['text']).select(range(500))
    test_set.set_format("torch")

    bmc_set = bmc_set.map(bgpt_tokenize_function, batched = True)
    bmc_set = bmc_set.remove_columns(['text'])
    bmc_set.set_format("torch")

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    
    training_args = TrainingArguments(output_dir="models/fine_tuning_bgpt", eval_strategy="epoch", run_name="run0_bgpt")

    trainer = Trainer(
        model=bgpt_model,
        args=training_args,
        train_dataset= train_set,
        eval_dataset= test_set,
        compute_metrics=compute_metrics,
    )

    results = trainer.train()

    
def fine_tune_cl_depricated(train_set, test_set, bmc_set):
    cl_tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-Longformer")
    cl_model = AutoModelForSequenceClassification.from_pretrained("yikuan8/Clinical-Longformer", num_labels= 4, problem_type="multi_label_classification")

    def cl_tokenize_function(examples):
        return cl_tokenizer(examples["text"], padding="max_length", truncation=True)

    train_set = train_set.map(cl_tokenize_function, batched = True)
    train_set = train_set.remove_columns(['text'])
    train_set.set_format("torch")

    test_set = test_set.map(cl_tokenize_function, batched = True)
    test_set = test_set.remove_columns(['text'])
    test_set.set_format("torch")
    
    bmc_set = bmc_set.map(cl_tokenize_function, batched = True)
    bmc_set = bmc_set.remove_columns(['text'])
    bmc_set.set_format("torch")
    
    train_dataloader = DataLoader(train_set, shuffle = True, batch_size = 5)
    eval_dataloader = DataLoader(test_set, batch_size = 8)

    optimizer = AdamW(cl_model.parameters(), lr=5e-3)
    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    cl_model.to(device)

    progress_bar = tqdm(range(num_training_steps))

    cl_model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = cl_model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    metric = evaluate.load("accuracy")
    cl_model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = cl_model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    metric.compute()


def fine_tune_bgpt_depricated(train_set, test_set, bmc_set):
    bgpt_tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
    bgpt_model = BioGptForSequenceClassification.from_pretrained("microsoft/biogpt", num_labels= 4, problem_type="multi_label_classification")

    def bgpt_tokenize_function(examples):
        return bgpt_tokenizer(examples["text"], padding="max_length", truncation=True)

    train_set = train_set.map(bgpt_tokenize_function, batched = True)
    train_set = train_set.remove_columns(['text'])
    train_set.set_format("torch")

    test_set = test_set.map(bgpt_tokenize_function, batched = True)
    test_set = test_set.remove_columns(['text'])
    test_set.set_format("torch")

    bmc_set = bmc_set.map(bgpt_tokenize_function, batched = True)
    bmc_set = bmc_set.remove_columns(['text'])
    bmc_set.set_format("torch")
    
    train_dataloader = DataLoader(train_set, shuffle = True, batch_size = 8)
    eval_dataloader = DataLoader(test_set, batch_size = 8)

    optimizer = AdamW(bgpt_model.parameters(), lr=5e-5)
    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    bgpt_model.to(device)

    progress_bar = tqdm(range(num_training_steps))

    bgpt_model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = bgpt_model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    metric = evaluate.load("accuracy")
    bgpt_model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = bgpt_model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    metric.compute()


if __name__ == "__main__":
    train_OLTV, train_targets, test_OLTV, test_targets, bmc_OLTV, bmc_targets = get_cleaned_data()

    train_set, test_set, bmc_set = get_sets_from_OLTV(train_OLTV, train_targets, test_OLTV, test_targets, bmc_OLTV, bmc_targets)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = AutoModelForMaskedLM.from_pretrained("yikuan8/Clinical-Longformer").to(device)
    tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-Longformer")

    def tokenize_function(batched_data):
        result = tokenizer(batched_data['text'], padding = 'max_length', truncation = True, max_length = 1024)
        if tokenizer.is_fast:
            result['word_ids'] = [result.word_ids(i) for i in range(len(result['input_ids']))]
        return result
    
    tokenized_train_set = train_set.map(tokenize_function, batched = True, remove_columns = ['text', 'labels'])
    tokenized_test_set = test_set.map(tokenize_function, batched = True, remove_columns = ['text','labels'])
    chunk_size = 1024

    def group_texts(batched_data):
        concatenated_examples = {k: sum(batched_data[k], []) for k in batched_data.keys()}
        total_length = len(concatenated_examples[list(batched_data.keys())[0]])
        total_length = (total_length // chunk_size) * chunk_size
        result = {k: [t[i: i+chunk_size] for i in range (0, total_length, chunk_size)] for k, t in concatenated_examples.items()}
        result['labels'] = result['input_ids'].copy()
        return result
    
    lm_train_set = tokenized_train_set.map(group_texts, batched = True)
    lm_test_set = tokenized_test_set.map(group_texts, batched = True)
    
    data_collator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm_probability = 0.15)

    batch_size = 4
    logging_steps = len(lm_train_set) // batch_size

    training_args = TrainingArguments(
        output_dir = "models/fine_tuning_cl",
        overwrite_output_dir = False,
        evaluation_strategy = "epoch", 
        num_train_epochs = 7,
        learning_rate = 2e-5,
        weight_decay = 0.01,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size = batch_size,
        push_to_hub = False,
        fp16 = True,
        logging_steps=logging_steps,
    )

    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = lm_train_set,
        eval_dataset = lm_test_set,
        data_collator = data_collator,
    )

    trainer.train()

    eval_results = trainer.evaluate()
    print(f"Loss: {eval_results['eval_loss']:.3f}")

    trainer.save_model()
    
    #fine_tune_bgpt(train_set, test_set, bmc_set)

    print('done')


    
        




        




