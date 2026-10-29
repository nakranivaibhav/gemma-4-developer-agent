"""usage: envs/harness/bin/python -I analysis/prefix_reuse.py runs/<prefix>/requests.jsonl.gz <n_pairs>
Token-level prefix reuse between an agent's consecutive requests, as vLLM would see it.
For a sample of requests whose parent (previous turn of the same conversation) is known:
  child_tok  = rendered prompt tokens of the request
  lcp_prompt = shared token prefix with the parent's rendered prompt          -> reuse if only the prompt is cached
  gen_ok     = whether the re-rendered parent turn (thinking + tool call) equals parent prompt + the tokens the model
               actually generated, i.e. whether the parent's generated tokens can be reused too
Prints the structural recompute share to compare with the server's prefix-cache miss share."""
import gzip,json,hashlib,random,sys,collections
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained('data/external/model')
H=lambda o: hashlib.md5(json.dumps(o,sort_keys=True).encode()).hexdigest()
path=sys.argv[1]; N=int(sys.argv[2])
full={}; pairs=[]
for i,l in enumerate(gzip.open(path,'rt')):
    try: r=json.loads(l)
    except Exception: break
    if r.get('status')!=200: continue
    m=r['request']['messages']
    hs=[H(m[:j]) for j in range(1,len(m)+1)]
    par=max((j for j in range(1,len(m)) if hs[j-1] in full), default=0)
    if par and len(m)-par==2: pairs.append((full[hs[par-1]],i))
    full[hs[-1]]=i
random.seed(0); pairs=random.sample(pairs,min(N,len(pairs))); need={x for p in pairs for x in p}
reqs={}
for i,l in enumerate(gzip.open(path,'rt')):
    if i in need:
        r=json.loads(l); reqs[i]=r
    if len(reqs)==len(need): break
def render(r):
    q=r['request']
    return tok.apply_chat_template(q['messages'],tools=q.get('tools'),add_generation_prompt=True,tokenize=False,**q.get('chat_template_kwargs',{}))
S=collections.Counter(); chk=[]
for p,c in pairs:
    tp,tc=render(reqs[p]),render(reqs[c])
    ip=tok.encode(tp,add_special_tokens=False); ic=tok.encode(tc,add_special_tokens=False)
    u=reqs[c]['response'].get('usage') or {}
    chk.append((len(ic),u.get('prompt_tokens')))
    L=0
    for a,b in zip(ip,ic):
        if a!=b: break
        L+=1
    gen=(reqs[p]['response'].get('usage') or {}).get('completion_tokens') or 0
    S['child']+=len(ic); S['lcp_prompt']+=L; S['parent_full_prefix']+=(L==len(ip)); S['n']+=1
    S['parent_gen']+=gen
    S['tool_and_turn']+=len(ic)-len(ip)   # tokens after the parent prompt: re-rendered turn + tool output
    if L<len(ip) and len(chk)<=400 and S['div_shown']<3:
        S['div_shown']+=1
        print('DIVERGE at',L,'of',len(ip),'| parent:',repr(tok.decode(ip[max(0,L-15):L+25]))[:300],'| child:',repr(tok.decode(ic[max(0,L-15):L+25]))[:300])
print(dict(S))
d=[a-b for a,b in chk if b]; print('render check: rendered minus server prompt_tokens, median',sorted(d)[len(d)//2],'min',min(d),'max',max(d))
n=S['child']
print(f"recompute if only the parent PROMPT is cached: {(n-S['lcp_prompt'])/n:.1%}")
print(f"recompute if the parent's generated tokens are cached too (lower bound): {(n-S['lcp_prompt']-S['parent_gen'])/n:.1%}")
print(f"parent prompt fully reused in {S['parent_full_prefix']}/{S['n']} pairs")
