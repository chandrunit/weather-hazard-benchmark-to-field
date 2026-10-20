# Cost-ratio sensitivity of the cost-aware threshold over 100 EV3 re-splits (probe on pool, threshold chosen on calibration part)
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
U='/mnt/user-data/uploads/anandha/results'
F=np.load(U+'/ev3_generalisation/ev3_features_fp16.npz'); bl=pd.read_csv(U+'/ev3_generalisation/ev3_blocks.csv'); Y=bl.y.values; N=len(Y)
RS=[1,2,3,5,10,20,50]; ths=np.linspace(0.01,0.99,99); rows=[]
def splits(groups,seed):
    fold=np.zeros(N,int)
    for f,(_,te) in enumerate(StratifiedGroupKFold(10,shuffle=True,random_state=seed).split(np.zeros(N),Y,groups)): fold[te]=f
    return np.where(fold>=5)[0],np.where((fold>=3)&(fold<5))[0],np.where(fold<3)[0]
for scheme,grp,base in [('random_grouped',bl.group.values,100),('sequence_blocked',bl.block.values,200)]:
    for rep in range(50):
        tr,ca,te=splits(grp,base+rep)
        for key in F.files:
            X=F[key].astype(np.float32); sc=StandardScaler().fit(X[tr]); c=LogisticRegression(C=0.5,max_iter=5000).fit(sc.transform(X[tr]),Y[tr])
            pc=c.predict_proba(sc.transform(X[ca]))[:,1]; pt=c.predict_proba(sc.transform(X[te]))[:,1]
            for r in RS:
                cc=[r*((pc<t)&(Y[ca]==1)).sum()+((pc>=t)&(Y[ca]==0)).sum() for t in ths]; t=ths[int(np.argmin(cc))]
                cost=lambda th:r*((pt<th)&(Y[te]==1)).sum()+((pt>=th)&(Y[te]==0)).sum()
                tb=1/(1+r)  # Bayes threshold applied to uncalibrated probe probabilities
                rows.append(dict(scheme=scheme,rep=rep,model=key,r=r,red=1-cost(t)/max(cost(0.5),1),red_bayes=1-cost(tb)/max(cost(0.5),1),
                                 miss=((pt<t)&(Y[te]==1)).sum()/(Y[te]==1).sum(),fa=((pt>=t)&(Y[te]==0)).sum()/(Y[te]==0).sum()))
D=pd.DataFrame(rows); D.to_csv('cost_ratio_sensitivity.csv',index=False)
g=D.groupby('r')
S=pd.DataFrame({'mean':g.red.mean(),'median':g.red.median(),'p_pos':g.red.apply(lambda x:(x>0).mean()),'lo':g.red.quantile(.025),'hi':g.red.quantile(.975),
                'model_mean_min':D.groupby(['r','model']).red.mean().groupby('r').min(),'model_mean_max':D.groupby(['r','model']).red.mean().groupby('r').max(),
                'bayes_mean':g.red_bayes.mean(),'miss':g.miss.mean(),'fa':g.fa.mean()}).round(3)
print(S.to_string()); S.to_csv('cost_ratio_sensitivity_summary.csv')
