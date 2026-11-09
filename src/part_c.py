from pathlib import Path
import sys, re, json, random, time
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import mock_llm
OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)
CATS={'billing','delivery','technical','fraud'}
MAX_INPUT_TOKENS=400
MAX_RETRIES=3
BASE_DELAY=0.05
EMAIL=re.compile(r'[\w.+-]+@[\w-]+\.[\w.]+')
PHONE=re.compile(r'\b[6-9]\d{9}\b')
ORDER=re.compile(r'ORD-\d{6}')
ZERO_PROMPT='''You are a careful support-ticket classifier. CLASSIFY each user message into exactly one category: billing, delivery, technical, fraud. Return JSON only in this form: {"category": "...", "reason": "..."}. Do not include markdown or extra text.'''
FEW_PROMPT='''You are a careful support-ticket classifier. CLASSIFY each user message into exactly one category: billing, delivery, technical, fraud. Return JSON only in this form: {"category": "...", "reason": "..."}. Do not include markdown or extra text.
EXAMPLES:
Ticket: I was charged twice and need the extra payment returned. Label: billing
Ticket: Someone made an unauthorized transaction on my account. Label: fraud
Ticket: The app crashes when I open the cart. Label: technical'''

def redact(text):
    counts={}
    text,n=EMAIL.subn('[EMAIL]',text); counts['email']=n
    text,n=PHONE.subn('[PHONE]',text); counts['phone']=n
    text,n=ORDER.subn('[ORDER_ID]',text); counts['order_id']=n
    assert not EMAIL.search(text) and not PHONE.search(text) and not ORDER.search(text)
    return text,counts

def call_retry(messages,ticket_id,log):
    for retry in range(MAX_RETRIES+1):
        try: return mock_llm.chat(messages)
        except mock_llm.TransientLLMError:
            if retry>=MAX_RETRIES: raise
            wait=BASE_DELAY*(2**retry)*random.uniform(.9,1.15)
            log.append({'ticket_id':ticket_id,'retry_number':retry,'wait_seconds':wait})
            print(f"retry ticket={ticket_id} retry={retry} wait={wait:.5f}s")
            time.sleep(wait)
    raise RuntimeError('unreachable')

def classify_run(prompt, budget=None):
    raw=pd.read_csv(ROOT/'data/raw/tickets.csv',keep_default_na=False)
    mock_llm.reset(); random.seed(42)
    rows=[]; retry_log=[]; total_tokens=0; pii={'email':0,'phone':0,'order_id':0}
    for t in raw.itertuples(index=False):
        text=t.text
        if not text.strip():
            rows.append({'ticket_id':t.ticket_id,'customer_id':t.customer_id,'status':'skipped_empty','category':'','tokens_used':0,'true_category':t.true_category}); continue
        est=mock_llm.est_tokens(text)
        if est>MAX_INPUT_TOKENS:
            rows.append({'ticket_id':t.ticket_id,'customer_id':t.customer_id,'status':'flagged_long','category':'','tokens_used':0,'true_category':t.true_category}); continue
        safe,counts=redact(text)
        for k,v in counts.items(): pii[k]+=v
        # The user message is exactly the redacted ticket text, with no wrapper.
        assert safe == re.sub(r'\s+$','',safe)
        assert not EMAIL.search(safe) and not PHONE.search(safe) and not ORDER.search(safe)
        if budget is not None and total_tokens>=budget:
            rows.append({'ticket_id':t.ticket_id,'customer_id':t.customer_id,'status':'budget_exhausted','category':'','tokens_used':0,'true_category':t.true_category}); continue
        response=call_retry([{'role':'system','content':prompt},{'role':'user','content':safe}],t.ticket_id,retry_log)
        used=response.usage.total_tokens; total_tokens+=used
        msg=response.choices[0].message; body=msg.content or ''
        status='flagged_invalid_output'; category=''
        try:
            parsed=json.loads(body)
            if parsed.get('category') in CATS:
                category=parsed['category']; status='classified_json'
        except (json.JSONDecodeError,AttributeError):
            words=re.findall(r'[a-z]+',body.lower()); found=CATS.intersection(words)
            if len(found)==1: category=next(iter(found)); status='classified_keyword'
        rows.append({'ticket_id':t.ticket_id,'customer_id':t.customer_id,'status':status,'category':category,'tokens_used':used,'true_category':t.true_category})
    df=pd.DataFrame(rows)
    classified=df[df.status.isin(['classified_json','classified_keyword'])]
    acc=float((classified.category==classified.true_category).mean()) if len(classified) else 0
    metrics={'total_tickets':len(df),'classified':len(classified),'accuracy':acc,'correct':int((classified.category==classified.true_category).sum()),'total_tokens':total_tokens,'tokens_per_classified':total_tokens/len(classified) if len(classified) else 0,'status_counts':df.status.value_counts().to_dict(),'pii_substitutions':pii,'pii_total':sum(pii.values()),'retry_count':len(retry_log),'retry_log':retry_log}
    return df,metrics

def route_results(df):
    seg=pd.read_csv(OUT/'customer_segments.csv').set_index('customer_id')
    routes=[]
    for r in df.itertuples(index=False):
        segment_name=''; premium=False
        if r.customer_id in seg.index:
            segment_name=seg.loc[r.customer_id,'segment_name']; premium=bool(seg.loc[r.customer_id,'is_premium'])
        if r.status in {'skipped_empty','flagged_long','flagged_invalid_output','budget_exhausted'}: route,priority='human_review','NORMAL'
        elif r.category=='fraud': route,priority='security_team','HIGH'
        elif premium: route,priority='premium_desk','HIGH'
        else: route,priority='support_queue','NORMAL'
        routes.append({'ticket_id':r.ticket_id,'customer_id':r.customer_id,'status':r.status,'category':r.category,'segment_name':segment_name,'priority':priority,'route':route,'tokens_used':r.tokens_used})
    return pd.DataFrame(routes)

def run():
    # PII substitution totals are measured once across all tickets; the two full runs are independent.
    zero,zm=classify_run(ZERO_PROMPT)
    few,fm=classify_run(FEW_PROMPT)
    # The committed final routing file is a separate few-shot run with its required budget.
    few_final,final_metrics=classify_run(FEW_PROMPT,budget=100000)
    # Confusion table only for validly classified zero-shot tickets.
    conf=pd.crosstab(zero.loc[zero.status.isin(['classified_json','classified_keyword']),'true_category'],zero.loc[zero.status.isin(['classified_json','classified_keyword']),'category']).reindex(index=['billing','delivery','technical','fraud'],columns=['billing','delivery','technical','fraud'],fill_value=0)
    # Budgeted zero-shot demo, reset again as an independent full run.
    bud,bm=classify_run(ZERO_PROMPT,budget=6000)
    bud.to_csv(OUT/'triage_budget_demo.csv',index=False)
    final=route_results(few_final); final.to_csv(OUT/'triage_results.csv',index=False)
    # The full-run PII totals are deterministic and identical across runs; retain first-run counts.
    summary={'zero_shot':zm,'few_shot':fm,'few_shot_final_budget':{'budget':100000,'total_tokens':final_metrics['total_tokens'],'budget_exhausted':int((few_final.status=='budget_exhausted').sum())},'zero_shot_confusion':conf.to_dict(orient='index'),'budget_demo':{'budget':6000,'budget_exhausted':int((bud.status=='budget_exhausted').sum()),'total_tokens':bm['total_tokens'],'status_counts':bm['status_counts']}}
    (OUT/'part_c_metrics.json').write_text(json.dumps(summary,indent=2))
    print('Part C complete:',zm['status_counts'],fm['accuracy'], 'budget exhausted',summary['budget_demo']['budget_exhausted'])
if __name__=='__main__': run()
