from pathlib import Path
import sqlite3, json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/raw'; CLEAN=ROOT/'data/clean'; OUT=ROOT/'outputs'
CLEAN.mkdir(parents=True,exist_ok=True); OUT.mkdir(exist_ok=True)

def run():
    customers=pd.read_csv(RAW/'customers.csv', keep_default_na=False)
    products=pd.read_csv(RAW/'products.csv')
    orders=pd.read_csv(RAW/'orders.csv', keep_default_na=False)
    # Customer cleaning
    city_before=customers['city'].astype(str).copy()
    customers['city']=customers['city'].astype(str).str.strip().str.title()
    city_changed=int((city_before!=customers['city']).sum())
    gender_before=customers['gender'].astype(str).copy()
    g=customers['gender'].astype(str).str.strip().str.lower()
    customers['gender']=g.map({'m':'male','male':'male','f':'female','female':'female'}).fillna('unknown')
    customers['age']=pd.to_numeric(customers['age'],errors='coerce')
    invalid_age=int((customers['age'].notna() & ~customers['age'].between(10,100)).sum())
    missing_age=int(customers['age'].isna().sum())
    customers.loc[~customers['age'].between(10,100),'age']=np.nan
    customers['age']=customers['age'].fillna(customers.groupby('city')['age'].transform('median'))
    customers['age']=customers['age'].fillna(customers['age'].median()).round().astype(int)
    gender_changed=int((gender_before!=customers['gender']).sum())
    customers['signup_date']=pd.to_datetime(customers['signup_date'])
    # Orders in the exact order specified in the brief
    orders['order_ts']=pd.to_datetime(orders['order_ts'])
    orders=orders.drop_duplicates()
    orders=orders.sort_values(['order_ts']).drop_duplicates('order_id',keep='last').copy()
    orders['quantity']=pd.to_numeric(orders['quantity'],errors='coerce')
    orders['discount_pct']=pd.to_numeric(orders['discount_pct'],errors='coerce')
    orders['order_amount']=pd.to_numeric(orders['order_amount'],errors='coerce')
    orders=orders.merge(products[['product_id','category','price_inr']],on='product_id',how='left')
    miss=orders['order_amount'].isna()
    orders.loc[miss,'order_amount']=(orders.loc[miss,'price_inr']*orders.loc[miss,'quantity']*(1-orders.loc[miss,'discount_pct']/100)).round(2)
    orders['delivery_minutes']=pd.to_numeric(orders['delivery_minutes'],errors='coerce')
    orders.loc[orders['delivery_minutes']<0,'delivery_minutes']=np.nan
    completed=orders['status'].eq('completed')
    completed_delivery_imputed=int(orders.loc[completed,'delivery_minutes'].isna().sum())
    med=orders.loc[completed].groupby('category')['delivery_minutes'].median()
    fill=orders.loc[completed,'category'].map(med)
    orders.loc[completed,'delivery_minutes']=orders.loc[completed,'delivery_minutes'].fillna(fill)
    # Store cleaned tables (strip helper columns from orders)
    orders=orders.drop(columns=['category','price_inr'])
    customers.to_csv(CLEAN/'customers.csv',index=False,date_format='%Y-%m-%d')
    products.to_csv(CLEAN/'products.csv',index=False)
    orders.to_csv(CLEAN/'orders.csv',index=False,date_format='%Y-%m-%d %H:%M:%S')
    db=ROOT/'data/instamart.db'
    if db.exists(): db.unlink()
    with sqlite3.connect(db) as con:
        customers.assign(signup_date=customers.signup_date.dt.strftime('%Y-%m-%d')).to_sql('customers',con,index=False)
        products.to_sql('products',con,index=False)
        orders.assign(order_ts=orders.order_ts.dt.strftime('%Y-%m-%d %H:%M:%S')).to_sql('orders',con,index=False)
    # Audits and checkpoint assertions
    raw_o=pd.read_csv(RAW/'orders.csv',keep_default_na=False)
    raw_c=pd.read_csv(RAW/'customers.csv',keep_default_na=False)
    raw_t=pd.read_csv(RAW/'tickets.csv',keep_default_na=False)
    raw_missing={}
    for label,frame in [('customers',raw_c),('products',products),('orders',raw_o)]:
        raw_missing[label]={'shape':list(frame.shape),'dtypes':{k:str(v) for k,v in frame.dtypes.items()},
            'missing':{k:{'count':int(frame[k].isna().sum()+((frame[k].astype(str).str.strip()=='').sum() if frame[k].dtype=='object' else 0)),
                          'percent':float((frame[k].isna().sum()+((frame[k].astype(str).str.strip()=='').sum() if frame[k].dtype=='object' else 0))/len(frame)*100)} for k in frame.columns}}
    raw_audit={
      'raw_orders_rows':int(len(raw_o)),'exact_duplicates':int(raw_o.duplicated().sum()),
      'pending_rows':int((raw_o.status=='pending').sum()),'distinct_order_ids':int(raw_o.order_id.nunique()),
      'non_numeric_order_amount':int(pd.to_numeric(raw_o.order_amount,errors='coerce').isna().sum()),
      'negative_delivery_minutes':int((pd.to_numeric(raw_o.delivery_minutes,errors='coerce')<0).sum()),
      'blank_delivery_minutes':int((raw_o.delivery_minutes.astype(str).str.strip()=='').sum()),
      'blank_age':int((raw_c.age.astype(str).str.strip()=='').sum()),
      'age_outside_10_100':int((pd.to_numeric(raw_c.age,errors='coerce').notna() & ~pd.to_numeric(raw_c.age,errors='coerce').between(10,100)).sum()),
      'distinct_raw_city_spellings':int(raw_c.city.nunique()),
      'distinct_nonblank_gender_spellings':int(raw_c.loc[raw_c.gender.astype(str).str.strip()!='','gender'].nunique()),
      'blank_gender':int((raw_c.gender.astype(str).str.strip()=='').sum())}
    audit={**raw_audit,'raw_shapes_dtypes_missing':raw_missing,
      'cleaning_changes':{'city_values_normalised':city_changed,'gender_values_normalised_or_unknown':gender_changed,'blank_ages_filled':missing_age,'out_of_range_ages_replaced':invalid_age,'exact_duplicate_rows_removed':int(raw_o.duplicated().sum()),'additional_duplicate_order_versions_removed':int(len(raw_o.drop_duplicates())-len(orders)),'order_amounts_reconstructed':int(pd.to_numeric(raw_o.order_amount,errors='coerce').isna().sum()),'negative_delivery_values_set_missing':raw_audit['negative_delivery_minutes'],'completed_delivery_values_imputed':completed_delivery_imputed},
      'clean_orders':int(len(orders)), 'completed':int((orders.status=='completed').sum()),
      'cancelled':int((orders.status=='cancelled').sum()),
      'completed_revenue':round(float(orders.loc[completed,'order_amount'].sum()),2),
      'customers':int(len(customers)), 'products':int(len(products)),
      'clean_city_counts':customers.city.value_counts().to_dict(),'clean_gender_counts':customers.gender.value_counts().to_dict(),
      'missing_clean_age':int(customers.age.isna().sum()),
      'missing_completed_delivery':int(orders.loc[completed,'delivery_minutes'].isna().sum()),
      'missing_cancelled_delivery':int(orders.loc[~completed,'delivery_minutes'].isna().sum())}
    (OUT/'part_a_audit.json').write_text(json.dumps(audit,indent=2))
    assert audit['clean_orders']==12309 and audit['completed']==11110 and audit['cancelled']==1199
    assert audit['completed_revenue']==2998668.80
    assert audit['missing_completed_delivery']==0 and audit['missing_cancelled_delivery']==1199
    with sqlite3.connect(db) as con:
        qs=(ROOT/'sql/queries.sql').read_text().split(';')
        results={}
        for i,q in enumerate(qs,1):
            if q.strip(): results[f'Q{i}']=pd.read_sql_query(q,con).to_dict(orient='records')
        (OUT/'sql_results.json').write_text(json.dumps(results,indent=2,default=str))
        q6=pd.read_sql_query(results_sql_q6(),con) if False else pd.read_sql_query((ROOT/'sql/queries.sql').read_text().split(';')[5],con)
        sql_summary={'Q1_rows':len(results['Q1']),'Q2_rows':len(results['Q2']),'Q2_total':round(sum(x['revenue'] for x in results['Q2']),2),
                     'Q2_max':max(results['Q2'],key=lambda x:x['revenue']),'Q3_rows':len(results['Q3']),'Q4_rows':len(results['Q4']),
                     'Q5_rows':len(results['Q5']),'Q5_avg_customer_spend':round(float(pd.read_sql_query("SELECT AVG(total_spend) v FROM (SELECT customer_id,SUM(order_amount) total_spend FROM orders WHERE status='completed' GROUP BY customer_id)",con).iloc[0,0]),2),
                     'Q6_rank_diff_row_number':int((q6.rank_in_city!=q6.row_num).sum()),'Q7_rows':len(results['Q7']),'Q7_final_cumulative':results['Q7'][-1]['cumulative_revenue']}
        city_stats=pd.read_sql_query("SELECT c.city,SUM(o.order_amount) sum,AVG(o.order_amount) mean,COUNT(*) count FROM orders o JOIN customers c USING(customer_id) WHERE o.status='completed' GROUP BY c.city ORDER BY c.city",con)
        sql_summary['city_stats']=city_stats.to_dict(orient='records')
        pivot=pd.pivot_table(orders[orders.status.eq('completed')].merge(customers[['customer_id','city']],on='customer_id').merge(products[['product_id','category']],on='product_id'),index='city',columns='category',values='order_amount',aggfunc='sum',fill_value=0,margins=True)
        sql_summary['pivot_grand_total']=round(float(pivot.loc['All','All']),2)
        sql_summary['q6_city_leaders']=q6.sort_values(['city','rank_in_city','customer_id']).groupby('city').first().reset_index()[['city','customer_id','total_spend']].to_dict(orient='records')
        (OUT/'part_a_sql_summary.json').write_text(json.dumps(sql_summary,indent=2,default=str))
    # EDA summary
    x=orders.loc[completed,'order_amount']
    q1,q3=x.quantile([.25,.75]); iqr=q3-q1
    eda={'mean':float(x.mean()),'median':float(x.median()),'skew':float(x.skew()),'q1':float(q1),'q3':float(q3),
         'lower_fence':float(q1-1.5*iqr),'upper_fence':float(q3+1.5*iqr),
         'outliers_below':int((x<q1-1.5*iqr).sum()),'outliers_above':int((x>q3+1.5*iqr).sum()),
         'correlation':orders.loc[completed,['order_amount','discount_pct','quantity','delivery_minutes']].corr().to_dict()}
    (OUT/'part_a_eda.json').write_text(json.dumps(eda,indent=2))

if __name__=='__main__': run()
