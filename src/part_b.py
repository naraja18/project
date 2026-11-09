from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)

def run():
    o=pd.read_csv(ROOT/'data/clean/orders.csv',parse_dates=['order_ts'])
    c=pd.read_csv(ROOT/'data/clean/customers.csv',parse_dates=['signup_date'])
    done=o[o.status.eq('completed')].copy(); snap=pd.Timestamp('2025-10-01')
    rfm=done.groupby('customer_id').agg(last_order=('order_ts','max'),frequency=('order_id','nunique'),monetary=('order_amount','sum')).reset_index()
    rfm['recency']=(snap-rfm.last_order).dt.days; rfm=rfm.drop(columns='last_order')
    cols=['recency','frequency','monetary']; scaler=StandardScaler(); z=scaler.fit_transform(rfm[cols])
    ks=[]
    for k in range(2,9):
        km=KMeans(n_clusters=k,n_init=10,random_state=42).fit(z)
        ks.append({'k':k,'inertia':float(km.inertia_),'silhouette':float(silhouette_score(z,km.labels_))})
    k=4; km=KMeans(n_clusters=k,n_init=10,random_state=42).fit(z); rfm['segment_id']=km.labels_
    prof=rfm.groupby('segment_id').agg(size=('customer_id','size'),mean_recency=('recency','mean'),mean_frequency=('frequency','mean'),mean_monetary=('monetary','mean')).reset_index()
    # Stable business names based on profile rather than arbitrary cluster IDs
    order=prof.sort_values('mean_monetary',ascending=False).segment_id.tolist()
    profile_index=prof.set_index('segment_id')
    name_map={}
    for sid in prof.segment_id:
        if sid==order[0]: name_map[sid]='Premium Value'
        elif sid==int(prof.loc[prof.mean_frequency.idxmax(),'segment_id']): name_map[sid]='Frequent Essentials'
        elif sid==int(prof.loc[prof.mean_recency.idxmax(),'segment_id']): name_map[sid]='At-Risk Occasional'
        else: name_map[sid]='Regular Value'
    premium_id=order[0]
    rfm['segment_name']=rfm.segment_id.map(name_map); rfm['is_premium']=rfm.segment_id.eq(premium_id)
    pca=PCA(n_components=2).fit(z)
    rfm[['customer_id','recency','frequency','monetary','segment_id','segment_name','is_premium']].to_csv(OUT/'customer_segments.csv',index=False)
    # Lapse cohort: only pre-cutoff orders are used as features.
    cutoff=pd.Timestamp('2025-09-01'); pre=o[(o.order_ts<cutoff)].copy(); pre_done=pre[pre.status.eq('completed')]
    cohort=pre_done.customer_id.unique()
    post_done=done[(done.order_ts>=cutoff)&(done.order_ts<pd.Timestamp('2025-10-01'))]
    lapsed=set(cohort)-set(post_done.customer_id)
    feat=pre.groupby('customer_id').agg(all_orders=('order_id','nunique'),cancelled=('status',lambda s:(s=='cancelled').sum())).reset_index()
    b=pre_done.groupby('customer_id').agg(last_order=('order_ts','max'),frequency=('order_id','nunique'),monetary=('order_amount','sum'),avg_discount=('discount_pct','mean'),avg_delivery=('delivery_minutes','mean')).reset_index()
    b['recency']=(cutoff-b.last_order).dt.days; b=b.drop(columns='last_order')
    cats=pre_done.merge(pd.read_csv(ROOT/'data/clean/products.csv')[['product_id','category']],on='product_id').groupby('customer_id').category.agg(lambda s:s.mode().iloc[0]).rename('favourite_category')
    feat=feat.merge(b,on='customer_id').merge(cats,on='customer_id').merge(c[['customer_id','city','age','gender','signup_date']],on='customer_id')
    feat['cancellation_rate']=feat.cancelled/feat.all_orders
    feat['tenure_days']=(cutoff-feat.signup_date).dt.days
    feat['lapsed']=feat.customer_id.isin(lapsed).astype(int)
    feat=feat.drop(columns=['all_orders','cancelled','signup_date'])
    assert not feat.isna().any().any(), f"Missing features: {feat.columns[feat.isna().any()].tolist()}"
    y=feat.pop('lapsed'); ids=feat.pop('customer_id')
    X=feat.copy(); X['customer_id']=ids
    Xtr,Xte,ytr,yte=train_test_split(np.arange(len(X)),y,test_size=.2,stratify=y,random_state=42)
    num=['recency','frequency','monetary','avg_discount','avg_delivery','cancellation_rate','age','tenure_days']
    cat=['favourite_category','city','gender']
    prep=ColumnTransformer([('num',StandardScaler(),num),('cat',OneHotEncoder(drop='first',handle_unknown='ignore'),cat)])
    lr=Pipeline([('prep',prep),('model',LogisticRegression(max_iter=1000,random_state=42))])
    rf=Pipeline([('prep',prep),('model',RandomForestClassifier(n_estimators=100,random_state=42))])
    lrgs=GridSearchCV(lr,{'model__C':[.01,.1,1,10,100]},cv=5,scoring='f1',n_jobs=-1).fit(X.iloc[Xtr],y.iloc[Xtr])
    rfgs=GridSearchCV(rf,{'model__max_depth':[3,5,8],'model__min_samples_leaf':[5,10,20]},cv=5,scoring='f1',n_jobs=-1).fit(X.iloc[Xtr],y.iloc[Xtr])
    models={'logistic':lrgs.best_estimator_,'random_forest':rfgs.best_estimator_}
    metrics={}
    for nm,m in models.items():
        pred=m.predict(X.iloc[Xte]); metrics[nm]={'best_params':(lrgs.best_params_ if nm=='logistic' else rfgs.best_params_),'cv_f1':float(lrgs.best_score_ if nm=='logistic' else rfgs.best_score_),'confusion_matrix':confusion_matrix(y.iloc[Xte],pred).tolist(),'precision':float(precision_score(y.iloc[Xte],pred,zero_division=0)),'recall':float(recall_score(y.iloc[Xte],pred,zero_division=0)),'f1':float(f1_score(y.iloc[Xte],pred,zero_division=0))}
    selected=models['random_forest'] if metrics['random_forest']['f1']>=metrics['logistic']['f1'] else models['logistic']
    # Training-only random oversampling for the class-balance experiment.
    train_idx=np.array(Xtr); pos=train_idx[y.iloc[train_idx].to_numpy()==1]; neg=train_idx[y.iloc[train_idx].to_numpy()==0]
    rng=np.random.RandomState(42); extra=rng.choice(pos,size=len(neg)-len(pos),replace=True) if len(pos)<len(neg) else np.array([],dtype=int)
    balanced=np.concatenate([train_idx,extra]); bal_lr=Pipeline([('prep',prep),('model',LogisticRegression(max_iter=1000,random_state=42,C=lrgs.best_params_['model__C']))]).fit(X.iloc[balanced],y.iloc[balanced])
    balpred=bal_lr.predict(X.iloc[Xte])
    # Verify serialization round trip.
    joblib.dump(selected,OUT/'lapse_model.joblib'); loaded=joblib.load(OUT/'lapse_model.joblib')
    assert np.array_equal(selected.predict(X.iloc[Xte]),loaded.predict(X.iloc[Xte]))
    # RF importance and LR coefficient signs
    rf_model=models['random_forest']; feature_names=rf_model.named_steps['prep'].get_feature_names_out(); importances=rf_model.named_steps['model'].feature_importances_
    top=sorted(zip(feature_names,importances),key=lambda x:x[1],reverse=True)[:5]
    lr_model=models['logistic']; lr_names=lr_model.named_steps['prep'].get_feature_names_out(); coefs=lr_model.named_steps['model'].coef_[0]
    coef_idx=np.argsort(np.abs(coefs))[-3:][::-1]
    # Premium subset of untouched test set; explicitly post-hoc, per brief.
    seg=rfm.set_index('customer_id')
    test_ids=X.iloc[Xte].customer_id
    premiums=test_ids[test_ids.map(lambda cid: bool(seg.loc[cid,'is_premium']) if cid in seg.index else False)]
    if len(premiums):
        loc=X.set_index('customer_id').loc[premiums]
        probs=selected.predict_proba(loc)[:,1]
        actual=pd.Series(y.to_numpy(),index=X.customer_id)
        ret=pd.DataFrame({'customer_id':premiums.to_numpy(),'lapse_probability':probs,'actually_lapsed':[int(actual[cid]) for cid in premiums]}).sort_values('lapse_probability',ascending=False).head(10)
    else: ret=pd.DataFrame(columns=['customer_id','lapse_probability','actually_lapsed'])
    ret.to_csv(OUT/'retention_list.csv',index=False)
    summary={'rfm_customers':len(rfm),'rfm_recency_median':float(rfm.recency.median()),'rfm_recency_min':int(rfm.recency.min()),'rfm_recency_max':int(rfm.recency.max()),'rfm_frequency_mean':float(rfm.frequency.mean()),'rfm_frequency_max':int(rfm.frequency.max()),'rfm_monetary_mean':float(rfm.monetary.mean()),'rfm_monetary_max':float(rfm.monetary.max()),'kmeans':ks,'chosen_k':k,'segment_profiles':prof.to_dict(orient='records'),'segment_name_map':{str(a):b for a,b in name_map.items()},'pca_explained_variance_ratio':pca.explained_variance_ratio_.tolist(),'cohort_size':len(feat),'lapsed_count':int(y.sum()),'lapsed_rate':float(y.mean()),'train_size':len(Xtr),'test_size':len(Xte),'baseline_f1':0.0,'baseline_accuracy':float((y.iloc[Xte]==0).mean()),'scaler_training_frequency_mean':float(lrgs.best_estimator_.named_steps['prep'].named_transformers_['num'].mean_[1]),'full_cohort_frequency_mean':float(X.frequency.mean()),'metrics':metrics,'oversampled_logistic':{'precision':float(precision_score(y.iloc[Xte],balpred,zero_division=0)),'recall':float(recall_score(y.iloc[Xte],balpred,zero_division=0)),'f1':float(f1_score(y.iloc[Xte],balpred,zero_division=0))},'rf_top_features':[(str(a),float(b)) for a,b in top],'lr_coefficients':[(str(lr_names[i]),float(coefs[i])) for i in coef_idx],'retention_top10_true_lapsed':int(ret.actually_lapsed.sum()) if len(ret) else 0,'retention_list_rows':len(ret)}
    (OUT/'part_b_metrics.json').write_text(json.dumps(summary,indent=2,default=float))
    print('Part B complete; selected model test F1:',round(metrics['random_forest']['f1'] if selected is models['random_forest'] else metrics['logistic']['f1'],4))
if __name__=='__main__': run()
