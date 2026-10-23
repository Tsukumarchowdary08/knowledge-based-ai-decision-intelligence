"""Reproducible UCI Phishing Websites (ID 327) calibrated decision-policy experiment."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 20261023
COSTS = dict(miss=20., false_alarm=2., review=0.6)

def load_data(path="data/Training Dataset.arff"):
    raw, _ = arff.loadarff(path)
    df = pd.DataFrame(raw)
    for col in df:
        df[col] = pd.to_numeric(df[col].map(lambda x: x.decode() if isinstance(x, bytes) else x), errors="raise")
    target = next((c for c in df if c.lower() in ("result", "class", "label")), df.columns[-1])
    y = (df.pop(target).to_numpy() == -1).astype(int)
    assert len(df) == 11055 and df.shape[1] == 30 and set(np.unique(y)) == {0, 1}
    return df, y

def models():
    return {
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2500, random_state=SEED)),
        "Random forest": RandomForestClassifier(n_estimators=250, min_samples_leaf=2, n_jobs=-1, random_state=SEED),
        "Histogram boosting": HistGradientBoostingClassifier(max_iter=180, l2_regularization=1., random_state=SEED)
    }

def actions(p, budget, policy, costs=COSTS):
    p = np.asarray(p)
    a = np.where(p * costs["miss"] <= (1-p)*costs["false_alarm"], 0, 1)
    n = int(np.floor(len(p)*budget))
    if n == 0:
        return a
    if policy == "knowledge":
        score = np.minimum(p*costs["miss"], (1-p)*costs["false_alarm"]) - costs["review"]
        eligible = np.where(score > 0)[0]
    elif policy == "confidence":
        score = -np.abs(p-.5)
        eligible = np.arange(len(p))
    else:
        raise ValueError(policy)
    chosen = eligible[np.argsort(-score[eligible], kind="stable")[:n]]
    a[chosen] = 2
    return a

def evaluate(y, p, budget, policy, costs=COSTS):
    a = actions(p, budget, policy, costs)
    loss = np.where(a == 2, costs["review"], np.where(a == 0, y*costs["miss"], (1-y)*costs["false_alarm"]))
    return dict(policy=policy, budget=float(budget), review_rate=float(np.mean(a==2)),
                coverage=float(np.mean(a!=2)), cost_per_case=float(loss.mean()),
                miss_rate=float(np.mean((a==0)&(y==1))),
                false_alarm_rate=float(np.mean((a==1)&(y==0))))

def bootstrap(y, p, budget=.2, repetitions=1000):
    rng = np.random.default_rng(SEED)
    differences = []
    for _ in range(repetitions):
        ix = rng.integers(len(y), size=len(y))
        differences.append(evaluate(y[ix], p[ix], budget, "confidence")["cost_per_case"] -
                           evaluate(y[ix], p[ix], budget, "knowledge")["cost_per_case"])
    return np.quantile(differences, [.025, .5, .975]).tolist()

def main():
    output = Path("results")
    output.mkdir(exist_ok=True)
    X, y = load_data()
    Xdev, Xtest, ydev, ytest = train_test_split(X, y, test_size=.2, random_state=SEED, stratify=y)
    Xtrain, Xval, ytrain, yval = train_test_split(Xdev, ydev, test_size=.25, random_state=SEED, stratify=ydev)
    validation = []
    for name, model in models().items():
        fitted = CalibratedClassifierCV(model, method="sigmoid", cv=3).fit(Xtrain, ytrain)
        p = fitted.predict_proba(Xval)[:,1]
        validation.append(dict(model=name, validation_roc_auc=roc_auc_score(yval,p),
                               validation_pr_auc=average_precision_score(yval,p),
                               validation_brier=brier_score_loss(yval,p),
                               validation_log_loss=log_loss(yval,p)))
    tab1 = pd.DataFrame(validation).sort_values("validation_brier")
    tab1.to_csv(output/"table1_validation_models.csv", index=False)
    selected = tab1.iloc[0]["model"]
    fitted = CalibratedClassifierCV(models()[selected], method="sigmoid", cv=3).fit(Xdev, ydev)
    p = fitted.predict_proba(Xtest)[:,1]
    tab2 = pd.DataFrame([evaluate(ytest,p,b,policy) for b in np.linspace(0,.4,9)
                         for policy in ("confidence","knowledge")])
    tab2.to_csv(output/"table2_policy_tradeoffs.csv", index=False)
    fig, ax = plt.subplots(figsize=(9,3))
    labels = ["Website\nfeatures","Calibrated\nAI","Risk/cost\nknowledge","Allow / block\nor review"]
    for i,label in enumerate(labels):
        ax.text(i,0,label,ha="center",va="center",bbox=dict(boxstyle="round,pad=.6",fc="white",ec="black"))
        if i: ax.annotate("",xy=(i-.38,0),xytext=(i-.62,0),arrowprops=dict(arrowstyle="->"))
    ax.set_xlim(-.6,3.6); ax.set_ylim(-.6,.6); ax.axis("off")
    fig.savefig(output/"figure1_framework.pdf",bbox_inches="tight");plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for name, group in tab2.groupby("policy"):
        ax.plot(group.review_rate*100,group.cost_per_case,marker="o",label=name)
    ax.set(xlabel="Actual reviewed cases (%)",ylabel="Mean simulated cost per case")
    ax.legend();ax.grid(alpha=.2);fig.tight_layout()
    fig.savefig(output/"figure2_tradeoff.pdf");plt.close(fig)
    summary=dict(dataset="UCI Phishing Websites 327",samples=len(y),features=X.shape[1],
                 test_samples=len(ytest),selected_model=selected,
                 test_roc_auc=roc_auc_score(ytest,p),test_brier=brier_score_loss(ytest,p),
                 scenario_costs=COSTS,perfect_review_assumed=True,
                 improvement_at_20pct_bootstrap_95pct=bootstrap(ytest,p))
    (output/"summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__ == "__main__":
    main()
