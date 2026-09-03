import sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams["font.family"]="NanumGothic"; matplotlib.rcParams["axes.unicode_minus"]=False
import matplotlib.pyplot as plt, numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.decompx_trajectory_verifier.config import PLOTS, RES
R=RES/"phase_d1a"; P=PLOTS/"phase_d1a"; P.mkdir(parents=True, exist_ok=True)
V=["V0","V1","V2","V3","V4"]
LAB={"V0":"V0 A0\n최종귀속","V1":"V1 FULL\nL1-12","V2":"V2 EARLY\nL1-8","V3":"V3 LATE\nL9-12","V4":"V4 FINAL\nL12"}
COL=dict(zip(V,["#7f7f7f","#1f77b4","#2ca02c","#d62728","#ff7f0e"]))
PR=["seen_source","loso_wj","loso_ps","loso_qs"]; PRL=["seen","LOSO WJ","LOSO PS","LOSO QS"]

def bars(f, out, title):
    d=pd.read_csv(R/f); d=d[d.scope=="pooled"]
    g=d.groupby(["protocol","variant"]).auroc.agg(["mean","std"])
    fig,ax=plt.subplots(figsize=(9,4.5)); x=np.arange(len(PR))
    for i,v in enumerate(V):
        m=[g.loc[(p,v),"mean"] for p in PR]; s=[g.loc[(p,v),"std"] for p in PR]
        ax.bar(x+i*.16-.32, m, .15, yerr=s, capsize=2, color=COL[v], label=LAB[v].replace("\n"," "))
    ax.axhline(.5, ls=":", c="k"); ax.set_xticks(x); ax.set_xticklabels(PRL)
    ax.set_ylabel("FP-vs-TP AUROC"); ax.set_title(title); ax.set_ylim(0,1)
    ax.grid(axis="y", alpha=.3); ax.legend(fontsize=8, ncol=5, loc="upper center")
    plt.tight_layout(); plt.savefig(P/out, dpi=140); plt.close()

bars("natural_results.csv","natural_loso_comparison.png","자연 분포 test (5시드 평균, 오차막대=표준편차)")
bars("matched_results.csv","matched_loso_comparison.png","길이 매칭 test (AUPRC 기준선 0.5)")

# early vs late
fig,ax=plt.subplots(figsize=(8,4.5))
nat=pd.read_csv(R/"natural_results.csv"); nat=nat[nat.scope=="pooled"]
g=nat.groupby(["protocol","variant"]).auroc.mean()
x=np.arange(len(PR))
for v,lab,c in (("V2","V2 EARLY L1-8","#2ca02c"),("V3","V3 LATE L9-12","#d62728"),
                ("V0","V0 A0 기준선","#7f7f7f")):
    ax.plot(x,[g.loc[(p,v)] for p in PR],"o-",color=c,label=lab,lw=2)
ax.axhline(.5,ls=":",c="k"); ax.set_xticks(x); ax.set_xticklabels(PRL)
ax.set_ylabel("AUROC"); ax.set_title("EARLY 와 LATE 는 seen 에서 동일하지만 OOD 에서 갈린다")
ax.grid(alpha=.3); ax.legend()
plt.tight_layout(); plt.savefig(P/"early_vs_late.png",dpi=140); plt.close()

l=pd.read_csv(R/"low_tp_loss.csv")
fig,axes=plt.subplots(1,2,figsize=(12,4.2))
for ax,r in zip(axes,[0.01,0.05]):
    t=l[l.target_tp_loss==r].pivot_table(index="protocol",columns="variant",values="fp_recall")
    x=np.arange(len(PR))
    for i,v in enumerate(V):
        ax.bar(x+i*.16-.32,[t.loc[p,v] for p in PR],.15,color=COL[v],label=LAB[v].replace("\n"," "))
    ax.set_xticks(x); ax.set_xticklabels(PRL); ax.set_ylabel("FP Recall")
    ax.set_title(f"목표 TP 오판정률 {r*100:.0f}%"); ax.grid(axis="y",alpha=.3)
axes[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig(P/"low_tp_loss_comparison.png",dpi=140); plt.close()
print("그림 4개 저장 ->",P)
