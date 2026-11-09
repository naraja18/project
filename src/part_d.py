from pathlib import Path
import sys, json, time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs'; STATE=ROOT/'state'; STATE.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT)); import mock_llm
POL=ROOT/'data/policies'

def make_chunks(overlap=10):
    chunks=[]
    for path in sorted(POL.glob('*.txt')):
        words=path.read_text().split(); start=0
        while start<len(words):
            end=min(start+40,len(words)); chunks.append({'source':path.name,'text':' '.join(words[start:end])})
            if end>=len(words): break
            start=end-overlap
    return chunks

class Retriever:
    def __init__(self,overlap=10):
        self.chunks=make_chunks(overlap); self.vectorizer=TfidfVectorizer()
        self.matrix=self.vectorizer.fit_transform([c['text'] for c in self.chunks])
    def retrieve(self,query,k=3):
        q=self.vectorizer.transform([query]); scores=cosine_similarity(q,self.matrix)[0]
        ix=sorted(range(len(scores)),key=lambda i:scores[i],reverse=True)[:k]
        return [{'source':self.chunks[i]['source'],'score':float(scores[i]),'text':self.chunks[i]['text']} for i in ix]
    def answer(self,question):
        hits=self.retrieve(question,3)
        context='\n'.join(f"[{h['source']}] {h['text']}" for h in hits)
        user=f"CONTEXT:\n{context}\nQUESTION:\n{question}"
        response=mock_llm.chat([{'role':'system','content':'POLICY ASSISTANT: Answer only from the supplied context. If the answer is not present, say I could not find that information in the policy documents.'},{'role':'user','content':user}])
        return response.choices[0].message.content,[h['source'] for h in hits]

# Tools and descriptions are explicit and machine-readable as required.
TOOLS_SPEC=[
 {'name':'get_order','description':'Look up one order in the analytical SQLite database.','parameters':{'type':'object','properties':{'order_id':{'type':'string'}},'required':['order_id']}},
 {'name':'search_policy','description':'Retrieve the three most relevant policy chunks for a query.','parameters':{'type':'object','properties':{'query':{'type':'string'}},'required':['query']}},
 {'name':'calculate_refund','description':'Calculate the permitted refund for a completed order.','parameters':{'type':'object','properties':{'order_amount':{'type':'number'},'delivery_minutes':{'type':'number'}},'required':['order_amount','delivery_minutes']}},
 {'name':'issue_refund','description':'Record an automatically approved refund in the refund ledger.','parameters':{'type':'object','properties':{'order_id':{'type':'string'},'refund_inr':{'type':'number'}},'required':['order_id','refund_inr']}},
 {'name':'escalate_to_human','description':'Record a refund request requiring human approval.','parameters':{'type':'object','properties':{'order_id':{'type':'string'},'refund_inr':{'type':'number'}},'required':['order_id','refund_inr']}}
]
retriever=None
def get_order(order_id):
    import sqlite3
    with sqlite3.connect(ROOT/'data/instamart.db') as con:
        row=con.execute('SELECT order_id,status,order_amount,delivery_minutes FROM orders WHERE order_id=?',(order_id,)).fetchone()
    if not row:return {'error':'order not found'}
    return {'order_id':row[0],'status':row[1],'order_amount':float(row[2]),'delivery_minutes':None if row[3] is None else float(row[3])}
def search_policy(query):
    global retriever
    if retriever is None: retriever=Retriever(10)
    return {'results':retriever.retrieve(query,3)}
def calculate_refund(order_amount,delivery_minutes):
    amount=float(order_amount) if delivery_minutes is not None and float(delivery_minutes)>45 else 0.0
    return {'refund_inr':round(amount,2)}
def _ledger(tool,order_id,refund_inr):
    rec={'timestamp':time.strftime('%Y-%m-%dT%H:%M:%S'),'tool':tool,'order_id':order_id,'refund_inr':round(float(refund_inr),2)}
    with (OUT/'refund_ledger.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
    return {'refund_inr':round(float(refund_inr),2)}
def issue_refund(order_id,refund_inr):return _ledger('issue_refund',order_id,refund_inr)
def escalate_to_human(order_id,refund_inr):return _ledger('escalate_to_human',order_id,refund_inr)
TOOL_MAP={'get_order':get_order,'search_policy':search_policy,'calculate_refund':calculate_refund,'issue_refund':issue_refund,'escalate_to_human':escalate_to_human}

class SimulatedCrash(Exception): pass

def _checkpoint(run_id,scenario_id,messages,tokens_used,step,executed,step_records,llm_calls):
    data={'run_id':run_id,'scenario_id':scenario_id,'messages':messages,'tokens_used':tokens_used,'step':step,'tools_executed':executed,'step_records':step_records,'llm_calls':llm_calls}
    (STATE/f'{run_id}.json').write_text(json.dumps(data,indent=2,default=str))

def _continue_agent(state, max_iterations=6, token_limit=6000,time_limit_seconds=10,crash_after_tools=None):
    run_id=state['run_id']; scenario_id=state['scenario_id']; messages=state['messages']; tokens_used=state['tokens_used']; step=state['step']; executed=state['tools_executed']; step_records=state.get('step_records',[]); start=time.monotonic(); llm_calls=state.get('llm_calls',0)
    while llm_calls<max_iterations:
        if time.monotonic()-start>=time_limit_seconds:
            return {'status':'FALLBACK_TIMEOUT','final_answer':'FALLBACK_TIMEOUT: hand off to a human agent; time limit reached.','steps':step_records,'step_count':step,'llm_calls':llm_calls,'tokens_used':tokens_used,'tools_executed':executed}
        response=mock_llm.chat(messages); llm_calls+=1; tokens_used+=response.usage.total_tokens
        if tokens_used>token_limit:
            return {'status':'FALLBACK_TOKEN_LIMIT','final_answer':'FALLBACK_TOKEN_LIMIT: hand off to a human agent; token budget exceeded.','steps':step_records,'step_count':step,'llm_calls':llm_calls,'tokens_used':tokens_used,'tools_executed':executed}
        msg=response.choices[0].message; calls=msg.tool_calls
        if not calls:
            answer=msg.content or ''
            return {'status':answer.split(':',1)[0],'final_answer':answer,'steps':step_records,'step_count':step,'llm_calls':llm_calls,'tokens_used':tokens_used,'tools_executed':executed}
        tool_call=calls[0]
        args=tool_call.arguments
        messages.append({'role':'assistant','content':None,'tool_calls':[{'id':tool_call.id,'name':tool_call.name,'arguments':args}]})
        observation=TOOL_MAP[tool_call.name](**args)
        executed.append(tool_call.name); step+=1
        messages.append({'role':'tool','tool_call_id':tool_call.id,'name':tool_call.name,'content':json.dumps(observation)})
        step_records.append({'tool':tool_call.name,'arguments':args,'observation':observation})
        state['step_records']=step_records
        _checkpoint(run_id,scenario_id,messages,tokens_used,step,executed,step_records,llm_calls)
        if crash_after_tools is not None and len(executed)==crash_after_tools:
            raise SimulatedCrash(f'Simulated crash after {crash_after_tools} tools')
    return {'status':'FALLBACK_MAX_ITERATIONS','final_answer':'FALLBACK_MAX_ITERATIONS: hand off to a human agent; maximum model iterations reached.','steps':step_records,'step_count':step,'llm_calls':llm_calls,'tokens_used':tokens_used,'tools_executed':executed}

def run_agent(user_message,scenario_id,max_iterations=6,token_limit=6000,time_limit_seconds=10,run_id=None,crash_after_tools=None):
    run_id=run_id or f'{scenario_id}_run'
    state={'run_id':run_id,'scenario_id':scenario_id,'messages':[{'role':'system','content':'AGENT: Use tools to inspect the order, retrieve refund policy and process the request safely.'},{'role':'user','content':user_message}],'tokens_used':0,'step':0,'tools_executed':[],'step_records':[],'llm_calls':0}
    return _continue_agent(state,max_iterations,token_limit,time_limit_seconds,crash_after_tools)

def resume_agent(run_id,max_iterations=6,token_limit=6000,time_limit_seconds=10):
    state=json.loads((STATE/f'{run_id}.json').read_text())
    return _continue_agent(state,max_iterations,token_limit,time_limit_seconds)

def eval_policy(ret):
    data=pd.read_csv(ROOT/'data/raw/policy_eval.csv'); ans=[]
    for r in data.itertuples(index=False):
        text,sources=ret.answer(r.question)
        ans.append({'question':r.question,'expected_doc':r.expected_doc,'expected_fact':r.expected_fact,'answer':text,'sources':sources})
    answerable=[x for x in ans if x['expected_doc']!='NONE']; noanswer=[x for x in ans if x['expected_doc']=='NONE']
    return {'expected_doc_top3':sum(x['expected_doc'] in x['sources'] for x in answerable),'answer_contains_fact':sum(x['expected_fact'] in x['answer'] for x in answerable),'unanswerable_fallback':sum(x['answer'].startswith('I could not find') for x in noanswer),'rows':ans}

def run():
    # Remove stale checkpoints so a clean reproduction has only this run's state.
    for old in STATE.glob('*.json'): old.unlink()
    # Policy RAG with overlap and no-overlap comparison
    global retriever
    retriever=Retriever(10)
    refund_top3=[x['source'] for x in retriever.retrieve('refund policy late delivery',3)]
    assert len(refund_top3)==3 and all(x=='refund_policy.txt' for x in refund_top3)
    rag=eval_policy(retriever)
    retriever0=Retriever(0); no=eval_policy(retriever0)
    # Restore overlapping retriever for agent tools.
    retriever=retriever
    (OUT/'policy_eval.json').write_text(json.dumps({'overlap_10':rag,'overlap_0':no},indent=2))
    # Scenario runs
    scenarios=pd.read_csv(ROOT/'data/raw/agent_scenarios.csv')
    ledger=OUT/'refund_ledger.jsonl'; ledger.write_text('')
    runs=[]
    for r in scenarios.itertuples(index=False):
        result=run_agent(r.message,r.scenario_id)
        record={'scenario_id':r.scenario_id,**result,'expected_outcome':r.expected_outcome,'expected_refund_inr':float(r.expected_refund_inr)}
        runs.append(record)
    # Guardrail demos
    s1=scenarios.iloc[0]
    token_demo=run_agent(s1.message,'S1_token_demo',token_limit=200)
    timeout_demo=run_agent(s1.message,'S1_timeout_demo',time_limit_seconds=0)
    # Crash after second tool, checkpoint then resume. Must not repeat prior tools.
    crash_id='S1_crash_resume'
    try: run_agent(s1.message,'S1_crash_resume',run_id=crash_id,crash_after_tools=2)
    except SimulatedCrash: pass
    resumed=resume_agent(crash_id)
    crash_tools=resumed['tools_executed']
    runs.extend([{'scenario_id':'S1_token_demo',**token_demo},{'scenario_id':'S1_timeout_demo',**timeout_demo},{'scenario_id':'S1_crash_resume',**resumed}])
    with (OUT/'agent_runs.jsonl').open('w') as f:
        for x in runs:f.write(json.dumps(x)+'\n')
    # Compact report
    report=[]
    for x in runs[:7]:
        refund=next((s['observation'].get('refund_inr') for s in x.get('steps_detail',[]) if 'observation' in s),None)
        # refund is encoded in the recorded tool observations; derive from expected scenario if needed.
        report.append({'scenario_id':x['scenario_id'],'status':x['status'],'expected_outcome':x['expected_outcome'],'expected_refund_inr':x['expected_refund_inr'],'llm_calls':x['llm_calls'],'tokens_used':x['tokens_used']})
    summary={'chunk_counts':pd.Series([c['source'] for c in make_chunks(10)]).value_counts().to_dict(),'chunks_total':len(make_chunks(10)),'refund_query_top3_sources':refund_top3,'overlap_10':{k:v for k,v in rag.items() if k!='rows'},'overlap_0':{k:v for k,v in no.items() if k!='rows'},'overlap_0_failed_questions':[x['question'] for x in no['rows'] if x['expected_doc']!='NONE' and (x['expected_doc'] not in x['sources'] or x['expected_fact'] not in x['answer'])],'scenario_results':report,'token_demo_status':token_demo['status'],'timeout_demo_status':timeout_demo['status'],'crash_resume_tools':crash_tools,'crash_resume_status':resumed['status'],'crash_resume_answer':resumed['final_answer']}
    (OUT/'part_d_metrics.json').write_text(json.dumps(summary,indent=2))
    print('Part D complete:',summary['chunk_counts'],summary['overlap_10'],summary['token_demo_status'],summary['timeout_demo_status'])
if __name__=='__main__': run()
