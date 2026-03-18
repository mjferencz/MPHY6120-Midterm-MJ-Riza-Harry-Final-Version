# MIDTERM PROJECT: Using CRISPR Dependency Scores to predict lineage and subtype of cancers
# Use Random Forest Categorical compared to most frequent and stratified baseline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE
import shap

# Part 1: Data Loading and Processing
def loading_features(crispr, lineage, subtype):
    """
    Inputs: Three individual dataframes
    Outputs: Two combined dataframes (one with gene data and lineage data,
                one with gene data and subtype data)
    """
    print("DATA LOADING AND FEATURES")
    print("=" * 30)
    print(f"The CRISPR Dependency Score file has {crispr.shape[0]} lines and their dependencies",
          f" across {crispr.shape[1] - 1} different genes, indicating the dependence of the cancer",
          f" on that gene for viability. The type of columns it has are {crispr.dtypes}")
    print(f"\nThe lineage file has {lineage.shape[0]} lines with {lineage["Lineage"].nunique()} different",
          f" lineages. The type of columns it has are {lineage.dtypes}")
    print(f"\n The subtype file has {subtype.shape[0]} lines with {subtype["OncotreeSubtype"].nunique()}",
          f" different subtypes. The type of columns it has are {subtype.dtypes}\n")
    
    # check for missing values and their relation to entire data size to determine impute strat
    dfs = {"CRISPR": crispr, "Lineage": lineage, "Subtype": subtype}
    for name, df in dfs.items():
        missing_values = df.drop(columns = ["line_id"]).isnull().sum()
        avg_missing = missing_values.mean()
        std_dev_missing = missing_values.std()
        print(f"\nIn the {name} dataset, there are {avg_missing} missing values per column on average with a standard deviation",
              f" of {std_dev_missing}. The average missing per column is {avg_missing / df.shape[0] * 100}% of the data.")
        if avg_missing / df.shape[0] * 100 == 0:
            print("No impute needed.")
        elif avg_missing / df.shape[0] * 100 < 5:
            print("This amount is small enough to allow for median impute strategy to fill the missing data.")
        else:
            print("This amount is too large for median impute strategy; must be decided more strategically.")
    
    # before impute, create the combined datasets (makes it so that we don't have to impute lineage
    # minimizes the number of imputes in total.)
    crispr_lineage = pd.merge(crispr, lineage, on = "line_id", how = "inner")
    cols_to_exclude = ['line_id', 'Lineage']
    cols_to_impute = [col for col in crispr_lineage.columns if col not in cols_to_exclude]
    for column in cols_to_impute:
        median = crispr_lineage[column].median()
        crispr_lineage[column] = crispr_lineage[column].fillna(median)
    print(f"\nAfter imputation for CRISPR + Lineage, there are {crispr_lineage.isnull().sum().sum()} missing values remaining,",
          " confirming that all missing data has been addressed.")
    
    crispr_subtype = pd.merge(crispr, subtype, on = "line_id", how = "inner")
    cols_to_exclude = ['line_id', 'OncotreeSubtype']
    cols_to_impute = [col for col in crispr_subtype.columns if col not in cols_to_exclude]
    for column in cols_to_impute:
        median = crispr_subtype[column].median()
        crispr_subtype[column] = crispr_subtype[column].fillna(median)
    print(f"\nAfter imputation for CRISPR + Subtype, there are {crispr_subtype.isnull().sum().sum()} missing values remaining,",
          " confirming that all missing data has been addressed.")
    
    # make sure sufficient datasize, otherwise it will create problems in training pipeline
    print("\n\nIn order to ensure that there is enough data to use to separate into training and testing",
          " we must exclude rare lineages and subtypes.")
    lineage_counts_before = crispr_lineage["Lineage"].value_counts()
    num_lineage_before = len(lineage_counts_before)
    num_samples_before = crispr_lineage.shape[0]
    print(f"Before dropping rare lineages:")
    print(f"  Number of lineages: {num_lineage_before}")
    print(f"  Total data points: {num_samples_before}")

    min_samples = 5
    counts = crispr_lineage['Lineage'].value_counts()
    valid_classes = counts[counts >= min_samples].index
    crispr_lineage = crispr_lineage[crispr_lineage['Lineage'].isin(valid_classes)]
    
    lineage_counts_after = crispr_lineage['Lineage'].value_counts()
    num_lineages_after = len(lineage_counts_after)
    num_samples_after = crispr_lineage.shape[0]
    print(f"\nAfter dropping rare lineages (min {min_samples} samples):")
    print(f"  Number of lineages: {num_lineages_after}")
    print(f"  Total data points: {num_samples_after}")
    
    subtypes_counts_before = crispr_subtype["OncotreeSubtype"].value_counts()
    num_subtype_before = len(subtypes_counts_before)
    num_samples_before = crispr_subtype.shape[0]
    print(f"\nBefore dropping rare subtypes:")
    print(f"  Number of subtypes: {num_subtype_before}")
    print(f"  Total data points: {num_samples_before}")

    min_samples = 5
    counts = crispr_subtype['OncotreeSubtype'].value_counts()
    valid_classes = counts[counts >= min_samples].index
    crispr_subtype = crispr_subtype[crispr_subtype['OncotreeSubtype'].isin(valid_classes)]
    
    subtypes_counts_after = crispr_subtype['OncotreeSubtype'].value_counts()
    num_subtype_after = len(subtypes_counts_after)
    num_samples_after = crispr_subtype.shape[0]
    print(f"\nAfter dropping rare subtypes (min {min_samples} samples):")
    print(f"  Number of subtypes: {num_subtype_after}")
    print(f"  Total data points: {num_samples_after}")
    
    return crispr_lineage, crispr_subtype

def corrs_distrs(crispr, lineage, subtype):
    """
    Inputs: Three individual dataframes for CRISPR data, lineage data, and subtype data
    Outputs: A histogram for each input dataframe, a heatmap for self gene correlations,
                a heatmap for lineage depedence on genes, a heatmap for subtype dependence on genes.
    """
    print("\n\nCORRELATIONS AND DISTRIBUTIONS")
    print("=" * 30)
    # start with crispr data. 500 genes is too large so we will use a subset of genes
    crispr_subset = crispr.drop(columns = ["line_id"]).sample(n = 50, axis = 1, random_state = 42)
    corr = crispr_subset.corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Gene Correlation Heatmap (Subset)")
    plt.savefig("outputs/gene_correlation_heatmap.png")
    # remove diagonal to get rid of one to one correlations between same genes
    corr_no_diag = crispr.drop(columns = ["line_id"]).corr().mask(np.eye(len(crispr.drop(columns = ["line_id"]).corr()), dtype=bool))
    avg_corr = corr_no_diag.mean().mean()
    print(f"Analysis of gene correlations: Taking a subset of the CRISPR dependency scores,",
          " there appears to be a few moderate correlations but the rest being minor. The moderate",
          " correlations are between genes of the same family, so it makes sense that they are correlated",
          " from an epigenetic viewpoint (genes of the same family are often in clusters and physical proximity",
          " should indicate overlap in rate of transcription) as well as a synthetic viewpoint (genes of the same",
          " family should work in combination)."
          f"Additionally, the average correlations among all the gene data is {avg_corr:.2f}",
          "which indicates that there is extremely low correlation between all the genes. This is helpful",
          "to show no indication of data leakage between genes as well as a varied dataset allowing",
          "better prediction for a varied population of patients.")
    # histogram of genes
    plt.clf()
    sns.histplot(crispr.drop(columns = ["line_id"]).values.flatten(), kde=True)
    plt.title("Distribution of CRISPR Dependency Scores")
    plt.xlabel("Dependency Score")
    plt.ylabel("Frequency")
    plt.savefig("outputs/crispr_histogram.png")
    print(f"\nAnalysis of CRISPR dependency score distribution: The overall distribution of scores",
        " has a negative (meaning the gene is more essential to viability) median with a short tail",
        " going leftwards, indicating most genes selected are important for cancer viability.",
        " This is expected and preferred as it is expected to be less arbitrarily correlated/linked",
        " to inhibitor effectiveness due to biological relevance.")

    # can't do individual correlations for lineage and subtype but can do correlation
    # with genes and do distributions
    plt.clf()
    plt.figure(figsize=(50,8))
    sns.histplot(lineage.drop(columns = ["line_id"]).values.flatten(), kde=True)
    plt.title("Distribution of Lineages")
    plt.xlabel("Lineage Names")
    plt.ylabel("Frequency")
    plt.savefig("outputs/lineage_histogram.png")
    print("\nAnalysis of lineage distribution: There are widely varying frequencies of different",
          " lineages which is to be expected since that is what is found in the real world too. There",
          " is still a sufficient amount of data for most lineages and those with little data would have",
          " been removed when creating the combined dataset. This is a viable dataset, considering using",
          " weighted metrics instead.")
    plt.clf()
    sns.histplot(subtype.drop(columns = ["line_id"]).values.flatten(), kde=True)
    plt.title("Distribution of Subtypes")
    plt.xlabel("")
    plt.ylabel("Frequency")
    plt.savefig("outputs/subtype_histogram.png")
    print("\nAnalysis of subtype distribution: There are also widely varying frequencies of different",
          " subtypes which is also expected. However, there are a lot of subtypes with little data even",
          " considering the removal when creating the combined dataset. It is therefore pertinent to consider",
          " data augmentation (see with and without that) as well as when considering relative metric scores.")

    X = crispr.drop(columns = ["line_id"])
    lin = lineage["Lineage"]
    lineage_means = X.groupby(lin).mean()
    plt.clf()
    plt.figure(figsize=(14,14))
    sns.heatmap(lineage_means, cmap="coolwarm", center=0)
    plt.title("Mean CRISPR Dependency by Cancer Lineage")
    plt.xlabel("Genes")
    plt.ylabel("Lineage")
    plt.savefig("outputs/crispr_lineage_dependency.png")
    print(f"\nAnalysis of gene dependencies for lineage: The heatmap shows the average CRISPR gene dependency",
          " scores across cancer lineages (all genes taken into account, only some names are able to be listed on axis due to spacing).",
          " Because stronger gene dependencies correspond to more negative values, darker blue regions show genes",
          " that are more universally essential for proliferation. The trend seems to be that there is a mixture of",
          " of no dependency, moderate dependency, and high dependency. This variation suggests that classification models",
          " could be created to leverage this, affirming the possibility of a model.")
    
    sub = subtype["OncotreeSubtype"]
    sub_means = X.groupby(sub).mean()
    plt.clf()
    plt.figure(figsize=(14,14))
    sns.heatmap(sub_means, cmap="coolwarm", center=0)
    plt.title("Mean CRISPR Dependency by Cancer Subtypes")
    plt.xlabel("Genes")
    plt.ylabel("Subtypes")
    plt.savefig("outputs/crispr_subtypes_dependency.png")
    print(f"\nAnalysis of gene dependencies for subtypes: Same analysis as for the lineage dependencies",
          " with a note that like for genes, all subtypes are used but not all names are shown due to spacing.")


# Part 2: Model Architecture and Data Preparation
def builder():
    """
    Input: None
    Output: Baseline model using stratified strategy and standard random forest model
    """
    # choose stratified strategy to take into account class imbalance and distribution
    baseline = DummyClassifier(strategy="stratified")
    # choose Random Forest due to non-linear gene interactions and effects, high dimensionality,
    # and biological noise resistant
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state = None
    )
    
    return baseline, rf

def prepare_fn(combined_df, target_col_name):
    """
    Input: Dataframe to use for data with imputation and name of target column
    Output: Training and testing data
    """
    gene_cols = [c for c in combined_df.columns if c not in ["line_id", target_col_name]]
    # features (X) are the CRISPR dependency scores
    X = combined_df[gene_cols]
    # target (Y) is the lineage or subtype
    Y = combined_df[target_col_name]
    
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size = 0.2, random_state = None, stratify = Y
    )
    
    return X_train, X_test, Y_train, Y_test

def prepare_fn_smote(combined_df, target_col_name):
    """
    Input: CRISPR Subtype combined dataset with imputation
    Output: Training and testing data with SMOTE applied to training (prevent leakage)
    """
    # features (X) are the CRISPR dependency scores
    gene_cols = [c for c in combined_df.columns if c not in ["line_id", target_col_name]]
    X = combined_df[gene_cols]
    # target (Y) is the subtype
    Y = combined_df[target_col_name]

    X_train_sub, X_test_sub, Y_train_sub, Y_test_sub = train_test_split(
        X, Y, test_size = 0.2, random_state = None, stratify = Y
    )
    # apply SMOTE to increase data size with augmented
    smote = SMOTE(
        sampling_strategy="not majority", k_neighbors = 1, random_state=42
    )
    X_train_sub, Y_train_sub = smote.fit_resample(X_train_sub, Y_train_sub)

    return X_train_sub, X_test_sub, Y_train_sub, Y_test_sub


# Part 3: Model Training and Metrics
def run_models(X_train, X_test, Y_train, Y_test, baseline_model, rf_model):
    """
    Inputs: Training data for features and target as well as test data, models
    Outputs: Result metrics for accuracy for single and top three, f1 scores, and roc-auc
    """
    results = {}
    models = {
        "Baseline": baseline_model,
        "Random Forest": rf_model
    }

    for name, model in models.items():
        model.fit(X_train, Y_train)
        preds = model.predict(X_test)
        
        acc = accuracy_score(Y_test, preds)
        f1 = f1_score(Y_test, preds, average='weighted')

        # sees if the top three guesses are correct rather than just the first (which is accuracy)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)
            top3 = top_k_accuracy_score(Y_test, proba, k=3, labels=model.classes_)
            # binarize from multi categorical for auc
            Y_test_bin = label_binarize(Y_test, classes=model.classes_)
            auc = roc_auc_score(Y_test_bin, proba, multi_class="ovr", average="weighted")
        else:
            top3 = None
            auc = None

        results[name] = {
            "accuracy": acc,
            "f1_weighted": f1,
            "top3_accuracy": top3,
            "roc_auc_one_vs_rest_weighted": auc
        }
    
    return pd.DataFrame(results)

def run_experiment_n_times(prepare_data_fn, data, target_col_name, n_runs = 10):
    """
    Input: Which data preparation to use (with or without SMOTE), the dataset, the target
            column name for prepare_data_fn, and number of runs
    Output: Dataframe of results
    """
    all_results = []
    for i in range(n_runs):
        X_train, X_test, Y_train, Y_test = prepare_data_fn(data, target_col_name)
        baseline, rf = builder()
        res = run_models(X_train, X_test, Y_train, Y_test, baseline, rf)
        res["Run"] = i
        all_results.append(res)
    results_concat = pd.concat(all_results)
    results_concat = results_concat.reset_index().rename(columns={"index": "metric"})
    avg_results = results_concat.groupby("metric")[["Baseline", "Random Forest"]].mean()
    std_results = results_concat.groupby("metric")[["Baseline", "Random Forest"]].std()
    avg_results["stat"] = "mean"
    std_results["stat"] = "std"
    summary_df = pd.concat([avg_results, std_results])
    
    return summary_df

# for visuals later
def reshape_experiment(df, experiment_name):
    """
    Input: Results dataframe and name of dataset used (lineage, subtype, subtype + SMOTE)
    Output: Merged dataframe for plotting purposes
    """
    df = df.copy()
    df = df.reset_index()
    df["experiment"] = experiment_name
    
    # split mean / std
    mean_df = df[df["stat"] == "mean"].drop(columns="stat")
    std_df  = df[df["stat"] == "std"].drop(columns="stat")
    
    # melt both
    mean_melt = mean_df.melt(id_vars=["metric", "experiment"], var_name="model", value_name="mean")
    std_melt  = std_df.melt(id_vars=["metric", "experiment"], var_name="model", value_name="std")
    merged = mean_melt.merge(std_melt, on=["metric", "experiment", "model"])
    
    return merged

# Main
if __name__ == "__main__":
    crispr_df = pd.read_csv("data/crispr_data.csv", index_col=0)
    lineage_df = pd.read_csv("data/lineage_data.csv", index_col=0)
    subtype_df = pd.read_csv("data/subtype_data.csv", index_col=0)
    crispr_lineage, crispr_subtype = loading_features(crispr_df, lineage_df, subtype_df)
    # corrs_distrs(crispr_df, lineage_df, subtype_df)
    # lineage first
    results_lin = run_experiment_n_times(prepare_fn, crispr_lineage, "Lineage")
    results_lin.rename(columns={'Baseline': 'Lineage Baseline', 'Random Forest': 'Lineage Random Forest'}, inplace=True)
    results_sub = run_experiment_n_times(prepare_fn, crispr_subtype, "OncotreeSubtype")
    results_sub.rename(columns={'Baseline': 'Subtype Baseline', 'Random Forest': 'Subtype Random Forest'}, inplace=True)
    results_sub_smote = run_experiment_n_times(prepare_fn_smote, crispr_subtype, "OncotreeSubtype")
    results_sub_smote.rename(columns={'Baseline': 'Subtype SMOTE Baseline', 'Random Forest': 'Subtype SMOTE Random Forest'}, inplace=True)
    print("METRIC RESULTS")
    print("=" * 30)
    print(results_lin)
    print("\n")
    print(results_sub)
    print("\n")
    print(results_sub_smote)
    
    # visual of results
    plot_df = pd.concat([
        reshape_experiment(results_lin, "Lineage"),
        reshape_experiment(results_sub, "Subtype"),
        reshape_experiment(results_sub_smote, "Subtype SMOTE")
    ])
    
    metrics = plot_df["metric"].unique()
    fig, axes = plt.subplots(2, 2, figsize=(14,10))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        subset = plot_df[plot_df["metric"] == metric]

        models = subset["model"].unique()
        experiments = subset["experiment"].unique()

        x_base = np.arange(len(experiments))
        width = 0.35

        for j, model in enumerate(models):
            model_data = subset[subset["model"] == model].set_index("experiment")

            # force consistent order
            model_data = model_data.reindex(experiments)

            means = model_data["mean"].values
            stds = model_data["std"].values

            x = x_base + j * width

            axes[i].bar(x, means, width=width, label=model)

            axes[i].errorbar(
                x,
                means,
                yerr=stds,
                fmt='none',
                capsize=4,
                color='black'
            )

        axes[i].set_xticks(x_base + width / 2)
        axes[i].set_xticklabels(experiments)

        axes[i].set_title(metric.replace("_", " ").title())
        axes[i].set_ylabel("Score")
        axes[i].set_xlabel("")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("outputs/results_with_errorbars.png")
    
    # Feature importance with SHAP
    # For Lineage
    X_train_lin, X_test_lin, Y_train_lin, Y_test_lin = prepare_fn(crispr_lineage, "Lineage")
    rf_lin = RandomForestClassifier(n_estimators=200, random_state=None)
    rf_lin.fit(X_train_lin, Y_train_lin)
    explainer_lin = shap.TreeExplainer(rf_lin)
    shap_values_lin = explainer_lin.shap_values(X_test_lin)
    feature_names = X_train_lin.columns
    if shap_values_lin.ndim == 3:  # multiclass
        shap_values_mean = np.abs(shap_values_lin).mean(axis=(0, 2))
    else:
        shap_values_mean = np.abs(shap_values_lin).mean(axis=0)
    feature_importance_lin = pd.DataFrame({'feature': list(feature_names), 'importance': shap_values_mean})
    feature_importance_lin = feature_importance_lin.sort_values('importance', ascending=False).head(20)
    plt.figure(figsize=(10,8))
    sns.barplot(data=feature_importance_lin, x='importance', y='feature')
    plt.title('Top 20 Feature Importances for Lineage Prediction (SHAP)')
    plt.savefig('outputs/shap_lineage_top20.png')
    
    # For Subtype
    X_train_sub, X_test_sub, Y_train_sub, Y_test_sub = prepare_fn(crispr_subtype, "OncotreeSubtype")
    rf_sub = RandomForestClassifier(n_estimators=200, random_state=None)
    rf_sub.fit(X_train_sub, Y_train_sub)
    explainer_sub = shap.TreeExplainer(rf_sub)
    shap_values_sub = explainer_sub.shap_values(X_test_sub)
    if shap_values_sub.ndim == 3:  # multiclass
        shap_values_mean_sub = np.abs(shap_values_sub).mean(axis=(0, 2))
    else:
        shap_values_mean_sub = np.abs(shap_values_sub).mean(axis=0)
    feature_importance_sub = pd.DataFrame({'feature': list(feature_names), 'importance': shap_values_mean_sub})
    feature_importance_sub = feature_importance_sub.sort_values('importance', ascending=False).head(20)
    plt.figure(figsize=(10,8))
    sns.barplot(data=feature_importance_sub, x='importance', y='feature')
    plt.title('Top 20 Feature Importances for Subtype Prediction (SHAP)')
    plt.savefig('outputs/shap_subtype_top20.png')
    
