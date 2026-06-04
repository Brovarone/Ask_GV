#!/usr/bin/env python3
"""
Pipeline avanzata multi-LLM per regolamenti GdR Markdown.
"""
from __future__ import annotations
import argparse, concurrent.futures as cf, csv, fnmatch, hashlib, json, os, random, re, subprocess, sys, tempfile, textwrap, time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import requests
APP_VERSION="2.0.0"
DEFAULT_TIMEOUT=180
DEFAULT_OUTPUT_ROOT=Path("output")
DEFAULT_MAX_WORKERS=4
DEFAULT_CHUNK_SIZE=2200
DEFAULT_CHUNK_OVERLAP=250
DEFAULT_TOP_K=8
DEFAULT_CONTEXT_CHARS=36000
DEFAULT_SUMMARY_MAX_CHARS=70000
DEFAULT_IGNORE_PATTERNS=[".git/*",".github/*","node_modules/*","venv/*",".venv/*","dist/*","build/*","site/*","*.png","*.jpg","*.jpeg","*.gif","*.webp","*.svg","*.pdf","*.zip","*.7z","*.mp3","*.mp4"]
PROFILE_PRESETS={"rules_lawyer":"Analizza come un rules lawyer rigoroso: coerenza, eccezioni, conflitti tra regole, edge case, ambiguita'.","systems_designer":"Analizza come un systems designer: bilanciamento, economia risorse, scaling, exploit, sinergie e anti-sinergie.","gm_experience":"Analizza come un game master: facilita' di gestione al tavolo, pacing, chiarezza operativa, riduzione attriti.","narrative_designer":"Analizza come un narrative designer: fantasy del personaggio, identita' meccanica, coerenza fiction-mechanics.","new_player":"Analizza come un playtester nuovo: onboarding, confusione probabile, punti oscuri, leggibilita'."}
SYSTEM_PROMPT="""Sei un assistente esperto di game design tabletop e analisi di regolamenti GdR.
Usa il materiale fornito come fonte primaria. Non inventare regole non supportate dal corpus.
Distingui chiaramente osservazioni, inferenze e proposte.
Se l'informazione e' insufficiente, dichiaralo esplicitamente.
Rispondi in italiano con questa struttura:
1. Lettura del problema
2. Cosa emerge dalle regole fornite
3. Criticita' / opportunita'
4. Proposta operativa
5. Impatti collaterali / trade-off
6. Test consigliati al tavolo
Quando possibile cita file, heading o chunk rilevanti.
"""
SUMMARY_PROMPT="""Sei un analista di regolamenti GdR. Estrai un summary tecnico ad alta densita' informativa.
Restituisci JSON valido con questa forma:
{
  "game_identity": "...",
  "core_loops": ["..."],
  "resolution_rules": ["..."],
  "combat_rules": ["..."],
  "progression_rules": ["..."],
  "resource_economy": ["..."],
  "classes_or_roles": ["..."],
  "magic_or_powers": ["..."],
  "conditions_and_status": ["..."],
  "ambiguities": ["..."],
  "keywords": ["..."],
  "summary_text": "..."
}
"""
JUDGE_SYSTEM_PROMPT="""Sei un valutatore tecnico di proposte di game design.
Valuta candidate response rispetto al corpus di regole fornito.
Premia: aderenza al corpus, chiarezza, impatto operativo, sensibilita' ai trade-off, utilita' per sviluppo.
Penalizza: allucinazioni, vaghezza, proposte non motivate, mancata considerazione degli effetti collaterali.
Restituisci JSON valido con winner, ranking e synthesis.
"""
@dataclass
class SourceDocument:
    path:str; title:str; content:str; sha256:str; chars:int; headings:List[str]=field(default_factory=list)
@dataclass
class Chunk:
    chunk_id:str; source_path:str; title:str; text:str; start_char:int; end_char:int; token_estimate:int; headings:List[str]=field(default_factory=list)
@dataclass
class RankedChunk:
    chunk:Chunk; score:float; reasons:List[str]=field(default_factory=list)
@dataclass
class Target:
    provider:str; model:str; profile:str; temperature:float=0.4; max_tokens:Optional[int]=None; enabled:bool=True; label:Optional[str]=None
@dataclass
class Answer:
    id:str; provider:str; model:str; profile:str; success:bool; latency_s:float; response_text:str; prompt_path:Optional[str]=None; error:Optional[str]=None; raw:Optional[Dict[str,Any]]=None

def ensure_dir(path:Path)->Path: path.mkdir(parents=True,exist_ok=True); return path
def write_text(path:Path,text:str)->None: path.write_text(text,encoding='utf-8')
def write_json(path:Path,data:Any)->None: path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
def append_jsonl(path:Path,data:Dict[str,Any])->None:
    with path.open('a',encoding='utf-8') as f: f.write(json.dumps(data,ensure_ascii=False)+'\n')
def read_text(path:Path)->str: return path.read_text(encoding='utf-8',errors='ignore')
def sha256_text(text:str)->str: return hashlib.sha256(text.encode('utf-8',errors='ignore')).hexdigest()
def slugify(value:str)->str: return re.sub(r'-+','-',re.sub(r'[^a-z0-9]+','-',value.lower().strip())).strip('-') or 'run'
def now_ts()->str: return time.strftime('%Y%m%d-%H%M%S')
def normalize_markdown(text:str)->str: return re.sub(r'\n{3,}','\n\n',text.replace('\r\n','\n').replace('\r','\n').replace('\t','    ')).strip()
def trim_text(text:str,max_chars:int)->str: return text if len(text)<=max_chars else text[:max_chars-3]+'...'
def estimate_tokens(text:str)->int: return max(1,len(text)//4)
def extract_headings(md:str)->List[str]: return [m.group(2).strip() for line in md.splitlines() if (m:=re.match(r'^(#{1,6})\s+(.*)$',line.strip()))]
def first_heading_or_filename(path:str,content:str)->str:
    hs=extract_headings(content); return hs[0] if hs else Path(path).stem
def safe_json_loads(text:str)->Optional[Any]:
    try: return json.loads(text)
    except Exception: pass
    for pat in [r'```json\s*(\{.*?\}|\[.*?\])\s*```', r'(\{.*\})']:
        m=re.search(pat,text,flags=re.S)
        if m:
            try: return json.loads(m.group(1))
            except Exception: pass
    return None
def retry_request(fn,retries:int=3,base_sleep:float=1.2):
    last=None
    for attempt in range(1,retries+1):
        try: return fn()
        except Exception as e:
            last=e
            if attempt==retries: break
            time.sleep(base_sleep*(2**(attempt-1))+random.uniform(0,0.4))
    raise last
def post_json(url:str,headers:Dict[str,str],payload:Dict[str,Any],timeout:int=DEFAULT_TIMEOUT)->Dict[str,Any]:
    r=requests.post(url,headers=headers,json=payload,timeout=timeout)
    try: data=r.json()
    except Exception: data={'text':r.text}
    if r.status_code>=400: raise RuntimeError(f"HTTP {r.status_code}: {json.dumps(data,ensure_ascii=False)[:2500]}")
    return data
def get_json(url:str,headers:Dict[str,str],timeout:int=DEFAULT_TIMEOUT)->Dict[str,Any]:
    r=requests.get(url,headers=headers,timeout=timeout)
    try: data=r.json()
    except Exception: data={'text':r.text}
    if r.status_code>=400: raise RuntimeError(f"HTTP {r.status_code}: {json.dumps(data,ensure_ascii=False)[:2500]}")
    return data
def extract_openai_like_text(data:Dict[str,Any])->str:
    try: return data['choices'][0]['message']['content'].strip()
    except Exception: pass
    try: return data['output_text'].strip()
    except Exception: pass
    return json.dumps(data,ensure_ascii=False)
def extract_gemini_text(data:Dict[str,Any])->str:
    try: return '\n'.join(part.get('text','') for part in data['candidates'][0]['content']['parts']).strip()
    except Exception: return json.dumps(data,ensure_ascii=False)
def iter_files_recursive(base:Path,ignore_patterns:List[str])->Iterable[Path]:
    for p in base.rglob('*'):
        if not p.is_file(): continue
        rel=p.relative_to(base).as_posix()
        if any(fnmatch.fnmatch(rel,patt) for patt in ignore_patterns): continue
        if p.suffix.lower()=='.md': yield p
def clone_repo(repo_url:str,workdir:Path)->Path:
    repo_dir=workdir/'repo'; subprocess.run(['git','clone','--depth','1',repo_url,str(repo_dir)],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE); return repo_dir
def expand_local_patterns(patterns:List[str])->List[Path]:
    out=[]
    for patt in patterns:
        matches=list(Path('.').glob(patt)) if any(ch in patt for ch in '*?[]') else [Path(patt)]
        for p in matches:
            if p.is_file() and p.suffix.lower()=='.md': out.append(p.resolve())
            elif p.is_dir(): out.extend(x.resolve() for x in p.rglob('*.md'))
    uniq=[]; seen=set()
    for p in out:
        s=str(p)
        if s not in seen: uniq.append(p); seen.add(s)
    return uniq
def load_documents_from_repo(repo_url:str,ignore_patterns:List[str])->List[SourceDocument]:
    with tempfile.TemporaryDirectory(prefix='rpg_repo_adv_') as tmp:
        repo_dir=clone_repo(repo_url,Path(tmp)); docs=[]
        for f in iter_files_recursive(repo_dir,ignore_patterns):
            rel=f.relative_to(repo_dir).as_posix(); content=normalize_markdown(read_text(f))
            if not content: continue
            docs.append(SourceDocument(rel,first_heading_or_filename(rel,content),content,sha256_text(content),len(content),extract_headings(content)))
        return docs
def load_documents_from_files(files:List[str])->List[SourceDocument]:
    docs=[]
    for f in expand_local_patterns(files):
        content=normalize_markdown(read_text(f))
        if not content: continue
        docs.append(SourceDocument(str(f),first_heading_or_filename(str(f),content),content,sha256_text(content),len(content),extract_headings(content)))
    return docs
def split_markdown_sections(text:str)->List[Tuple[str,str]]:
    sections=[]; heading='intro'; lines=[]
    for line in text.splitlines():
        m=re.match(r'^(#{1,6})\s+(.*)$',line.strip())
        if m:
            if lines: sections.append((heading,'\n'.join(lines).strip())); lines=[]
            heading=m.group(2).strip(); lines.append(line)
        else: lines.append(line)
    if lines: sections.append((heading,'\n'.join(lines).strip()))
    return [(h,s) for h,s in sections if s.strip()]
def chunk_document(doc:SourceDocument,chunk_size:int,overlap:int)->List[Chunk]:
    out=[]; idx=0
    for heading,sec in split_markdown_sections(doc.content):
        if len(sec)<=chunk_size:
            out.append(Chunk(f"{slugify(doc.path)}-{idx:04d}",doc.path,doc.title,sec,0,len(sec),estimate_tokens(sec),[heading])); idx+=1; continue
        start=0
        while start<len(sec):
            end=min(len(sec),start+chunk_size); window=sec[start:end]
            if end<len(sec):
                lb=max(window.rfind('\n\n'),window.rfind('\n'),window.rfind('. '))
                if lb>chunk_size//2: end=start+lb+1; window=sec[start:end]
            out.append(Chunk(f"{slugify(doc.path)}-{idx:04d}",doc.path,doc.title,window.strip(),start,end,estimate_tokens(window),[heading])); idx+=1
            if end>=len(sec): break
            start=max(0,end-overlap)
    return out
def build_keyword_counter(documents:List[SourceDocument])->Dict[str,int]:
    cnt={}
    for d in documents:
        for w in re.findall(r'[a-zA-ZÀ-ÿ][a-zA-ZÀ-ÿ0-9_-]{2,}',d.content.lower()): cnt[w]=cnt.get(w,0)+1
    return dict(sorted(cnt.items(),key=lambda kv:kv[1],reverse=True))
def query_terms(question:str)->List[str]:
    terms=re.findall(r'[a-zA-ZÀ-ÿ][a-zA-ZÀ-ÿ0-9_-]{2,}',question.lower())
    stop={'che','come','quale','quali','delle','della','dello','degli','nelle','nello','per','con','una','uno','del','dei','dai','alla','allo','dalla','dalle','sulle','sulla','sistema','regole','gioco','gdr','rpg'}
    return [t for t in terms if t not in stop]
def rank_chunks(question:str,chunks:List[Chunk],summary:Optional[Dict[str,Any]],top_k:int)->List[RankedChunk]:
    qset=set(query_terms(question)); sk=set((summary or {}).get('keywords',[])[:60]) if isinstance(summary,dict) else set(); ranked=[]
    for c in chunks:
        score=0.0; reasons=[]; tl=c.text.lower(); qhits=sum(tl.count(q) for q in qset)
        if qhits: score+=qhits*3.0; reasons.append(f'qhits={qhits}')
        hhits=sum(1 for h in c.headings if any(q in h.lower() for q in qset))
        if hhits: score+=hhits*4.0; reasons.append(f'hhits={hhits}')
        khits=sum(1 for kw in sk if kw and kw.lower() in tl)
        if khits: score+=min(khits,12)*0.4; reasons.append(f'khits={khits}')
        if any(w in tl for w in ['danno','spell','magic','mana','slot','classe','azione','turno','progress']): score+=1.2
        score+=min(c.token_estimate/500,2.0); ranked.append(RankedChunk(c,score,reasons))
    ranked.sort(key=lambda x:x.score,reverse=True); return ranked[:top_k]
def compose_context(summary:Optional[Dict[str,Any]],ranked_chunks:List[RankedChunk],max_chars:int)->str:
    parts=[]
    if summary: parts+=['## SUMMARY STRUTTURATO',json.dumps(summary,ensure_ascii=False,indent=2)]
    parts.append('## ESTRATTI RILEVANTI'); current='\n\n'.join(parts)
    for rc in ranked_chunks:
        block=f"\n\n### CHUNK {rc.chunk.chunk_id}\nsource_path: {rc.chunk.source_path}\ntitle: {rc.chunk.title}\nheadings: {', '.join(rc.chunk.headings)}\nscore: {rc.score:.2f}\ntext:\n{rc.chunk.text}\n"
        if len(current)+len(block)>max_chars: break
        current+=block
    return current
def build_summary_corpus(documents:List[SourceDocument],max_chars:int)->str:
    blocks=[]; current=0
    for d in documents:
        block=f"\n\n# FILE: {d.path}\nTITLE: {d.title}\n\n{trim_text(d.content,16000)}"
        if current+len(block)>max_chars: break
        blocks.append(block); current+=len(block)
    return ''.join(blocks).strip()
def build_user_prompt(question:str,summary:Dict[str,Any],chunks:List[Chunk],profile:str,top_k:int,max_chars:int)->str:
    ranked=rank_chunks(question,chunks,summary,top_k); context=compose_context(summary,ranked,max_chars); profile_instruction=PROFILE_PRESETS.get(profile,PROFILE_PRESETS['systems_designer'])
    return textwrap.dedent(f"""
    Contesto del progetto:
    Stai analizzando un corpus di regole GdR in Markdown gia' preprocessato.

    Profilo di risposta:
    {profile_instruction}

    Domanda di sviluppo:
    {question}

    Materiale di riferimento:
    {context}

    Vincoli:
    - usa il corpus fornito come base primaria
    - se manca informazione, dichiaralo
    - distingui osservazioni, inferenze e proposte
    - cita file, heading o chunk quando utile
    - proponi modifiche operative, non solo teoria astratta
    """).strip()
def openai_style_messages(user_prompt:str)->List[Dict[str,str]]: return [{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':user_prompt}]
def call_openrouter(model:str,api_key:str,messages:List[Dict[str,str]],temperature:float=0.4,max_tokens:Optional[int]=None,response_format:Optional[Dict[str,Any]]=None)->Dict[str,Any]:
    payload={'model':model,'messages':messages,'temperature':temperature}
    if max_tokens: payload['max_tokens']=max_tokens
    if response_format: payload['response_format']=response_format
    headers={'Authorization':f'Bearer {api_key}','Content-Type':'application/json','HTTP-Referer':'https://localhost','X-Title':'multi-llm-rpg-rules-advanced'}
    return retry_request(lambda: post_json('https://openrouter.ai/api/v1/chat/completions',headers,payload))
def call_perplexity(model:str,api_key:str,messages:List[Dict[str,str]],temperature:float=0.3,max_tokens:Optional[int]=None)->Dict[str,Any]:
    payload={'model':model,'messages':messages,'temperature':temperature}
    if max_tokens: payload['max_tokens']=max_tokens
    headers={'Authorization':f'Bearer {api_key}','Content-Type':'application/json'}
    return retry_request(lambda: post_json('https://api.perplexity.ai/chat/completions',headers,payload))
def call_nvidia(model:str,api_key:str,messages:List[Dict[str,str]],base_url:str,temperature:float=0.3,max_tokens:Optional[int]=None)->Dict[str,Any]:
    payload={'model':model,'messages':messages,'temperature':temperature}
    if max_tokens: payload['max_tokens']=max_tokens
    headers={'Authorization':f'Bearer {api_key}','Content-Type':'application/json'}
    return retry_request(lambda: post_json(base_url.rstrip('/')+'/chat/completions',headers,payload))
def call_gemini_generate_content(model:str,api_key:str,system_prompt:str,user_prompt:str,temperature:float=0.4)->Dict[str,Any]:
    payload={'systemInstruction':{'parts':[{'text':system_prompt}]},'contents':[{'role':'user','parts':[{'text':user_prompt}]}],'generationConfig':{'temperature':temperature}}
    return retry_request(lambda: post_json(f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}',{'Content-Type':'application/json'},payload))
def call_gemini_generate_with_file_uri(model:str,api_key:str,system_prompt:str,file_uri:str,mime_type:str,question_prompt:str,temperature:float=0.4)->Dict[str,Any]:
    payload={'systemInstruction':{'parts':[{'text':system_prompt}]},'contents':[{'role':'user','parts':[{'fileData':{'mimeType':mime_type,'fileUri':file_uri}},{'text':question_prompt}]}],'generationConfig':{'temperature':temperature}}
    return retry_request(lambda: post_json(f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}',{'Content-Type':'application/json'},payload))
def call_gemini_upload_file(api_key:str,file_path:Path,mime_type:str='text/markdown')->Dict[str,Any]:
    url=f'https://generativelanguage.googleapis.com/upload/v1beta/files?key={api_key}'; headers={'X-Goog-Upload-Protocol':'multipart'}; files={'metadata':(None,json.dumps({'display_name':file_path.name}),'application/json'),'file':(file_path.name,file_path.read_bytes(),mime_type)}
    def _do():
        r=requests.post(url,headers=headers,files=files,timeout=DEFAULT_TIMEOUT)
        try: data=r.json()
        except Exception: data={'text':r.text}
        if r.status_code>=400: raise RuntimeError(f"HTTP {r.status_code}: {json.dumps(data,ensure_ascii=False)[:2500]}")
        return data
    return retry_request(_do)
def generate_summary(provider:str,model:str,documents:List[SourceDocument],gemini_use_files:bool=False,work_dir:Optional[Path]=None)->Dict[str,Any]:
    corpus=build_summary_corpus(documents,DEFAULT_SUMMARY_MAX_CHARS); user_prompt='Analizza il seguente corpus Markdown relativo a un gioco di ruolo tabletop e costruisci un summary tecnico.\n\n'+corpus
    if provider=='openrouter':
        api_key=os.getenv('OPENROUTER_API_KEY');
        if not api_key: raise RuntimeError('OPENROUTER_API_KEY non impostata')
        raw=call_openrouter(model,api_key,openai_style_messages(user_prompt),response_format={'type':'json_object'}); return safe_json_loads(extract_openai_like_text(raw)) or {'summary_text':extract_openai_like_text(raw)}
    if provider=='gemini':
        api_key=os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
        if not api_key: raise RuntimeError('GEMINI_API_KEY / GOOGLE_API_KEY non impostata')
        if gemini_use_files and work_dir is not None:
            consolidated=work_dir/'gemini_summary_corpus.md'; write_text(consolidated,corpus); up=call_gemini_upload_file(api_key,consolidated,'text/markdown'); file_uri=up.get('file',{}).get('uri') or up.get('uri'); mime=up.get('file',{}).get('mimeType','text/markdown')
            if file_uri:
                raw=call_gemini_generate_with_file_uri(model,api_key,SUMMARY_PROMPT,file_uri,mime,'Analizza il file allegato e restituisci JSON valido secondo lo schema richiesto.'); return safe_json_loads(extract_gemini_text(raw)) or {'summary_text':extract_gemini_text(raw),'file_upload':up}
        raw=call_gemini_generate_content(model,api_key,SUMMARY_PROMPT,user_prompt); return safe_json_loads(extract_gemini_text(raw)) or {'summary_text':extract_gemini_text(raw)}
    raise RuntimeError(f'Provider summary non supportato: {provider}')
def build_knowledge_pack(documents:List[SourceDocument],cache_dir:Path,settings:Dict[str,Any])->Dict[str,Any]:
    ensure_dir(cache_dir); r=settings.get('retrieval',{}); chunk_size=r.get('chunk_size',DEFAULT_CHUNK_SIZE); overlap=r.get('chunk_overlap',DEFAULT_CHUNK_OVERLAP); chunks=[]
    for d in documents: chunks.extend(chunk_document(d,chunk_size,overlap))
    scfg=settings.get('summary',{})
    try: summary=generate_summary(scfg.get('provider','gemini'),scfg.get('model','gemini-2.5-pro'),documents,bool(scfg.get('gemini_use_files_api',False)),cache_dir)
    except Exception as e: summary={'summary_text':f'Summary automatica non disponibile: {e}','keywords':list(build_keyword_counter(documents).keys())[:60],'ambiguities':[]}
    stats={'documents':len(documents),'chunks':len(chunks),'total_chars':sum(d.chars for d in documents),'keywords_top':list(build_keyword_counter(documents).items())[:150]}
    manifest={'version':APP_VERSION,'created_at':now_ts(),'summary_provider':scfg.get('provider','gemini'),'summary_model':scfg.get('model','gemini-2.5-pro'),'documents':len(documents),'chunks':len(chunks)}
    write_json(cache_dir/'documents.json',[asdict(d) for d in documents]); write_json(cache_dir/'chunks.json',[asdict(c) for c in chunks]); write_json(cache_dir/'summary.json',summary); write_json(cache_dir/'stats.json',stats); write_json(cache_dir/'manifest.json',manifest); write_text(cache_dir/'corpus_full.md','\n\n'.join([f"# FILE: {d.path}\n\n{d.content}" for d in documents]))
    return {'documents':documents,'chunks':chunks,'summary':summary,'stats':stats,'manifest':manifest}
def load_knowledge_pack(cache_dir:Path)->Dict[str,Any]:
    documents=[SourceDocument(**x) for x in json.loads(read_text(cache_dir/'documents.json'))]; chunks=[Chunk(**x) for x in json.loads(read_text(cache_dir/'chunks.json'))]; summary=json.loads(read_text(cache_dir/'summary.json')); stats=json.loads(read_text(cache_dir/'stats.json')) if (cache_dir/'stats.json').exists() else {}; manifest=json.loads(read_text(cache_dir/'manifest.json')) if (cache_dir/'manifest.json').exists() else {}
    return {'documents':documents,'chunks':chunks,'summary':summary,'stats':stats,'manifest':manifest}
def ask_target(target:Target,question:str,kp:Dict[str,Any],run_dir:Path,settings:Dict[str,Any])->Answer:
    r=settings.get('retrieval',{}); user_prompt=build_user_prompt(question,kp['summary'],kp['chunks'],target.profile,r.get('top_k',DEFAULT_TOP_K),r.get('max_context_chars',DEFAULT_CONTEXT_CHARS)); prompt_path=ensure_dir(run_dir/'prompts')/f"{slugify(target.provider)}-{slugify(target.model)}-{slugify(target.profile)}.txt"; write_text(prompt_path,user_prompt); t0=time.time(); raw=None
    try:
        if target.provider=='openrouter':
            api_key=os.getenv('OPENROUTER_API_KEY');
            if not api_key: raise RuntimeError('OPENROUTER_API_KEY non impostata')
            raw=call_openrouter(target.model,api_key,openai_style_messages(user_prompt),target.temperature,target.max_tokens); text=extract_openai_like_text(raw)
        elif target.provider=='perplexity':
            api_key=os.getenv('PERPLEXITY_API_KEY');
            if not api_key: raise RuntimeError('PERPLEXITY_API_KEY non impostata')
            raw=call_perplexity(target.model,api_key,openai_style_messages(user_prompt),target.temperature,target.max_tokens); text=extract_openai_like_text(raw)
        elif target.provider=='gemini':
            api_key=os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
            if not api_key: raise RuntimeError('GEMINI_API_KEY / GOOGLE_API_KEY non impostata')
            raw=call_gemini_generate_content(target.model,api_key,SYSTEM_PROMPT,user_prompt,target.temperature); text=extract_gemini_text(raw)
        elif target.provider=='nvidia':
            api_key=os.getenv('NVIDIA_API_KEY') or os.getenv('NIM_API_KEY')
            if not api_key: raise RuntimeError('NVIDIA_API_KEY / NIM_API_KEY non impostata')
            base_url=settings.get('providers',{}).get('nvidia',{}).get('base_url','https://integrate.api.nvidia.com/v1'); raw=call_nvidia(target.model,api_key,openai_style_messages(user_prompt),base_url,target.temperature,target.max_tokens); text=extract_openai_like_text(raw)
        else: raise RuntimeError(f'Provider non supportato: {target.provider}')
        return Answer(f"{target.provider}:{target.model}:{target.profile}",target.provider,target.model,target.profile,True,time.time()-t0,text,str(prompt_path),None,raw)
    except Exception as e:
        return Answer(f"{target.provider}:{target.model}:{target.profile}",target.provider,target.model,target.profile,False,time.time()-t0,'',str(prompt_path),str(e),raw)
def run_judge(question:str,answers:List[Answer],kp:Dict[str,Any],settings:Dict[str,Any],run_dir:Path)->Optional[Dict[str,Any]]:
    j=settings.get('judge',{}); provider=j.get('provider','openrouter'); model=j.get('model'); candidates=[a for a in answers if a.success]
    if not j.get('enabled',False) or not model or len(candidates)<2: return None
    ranked=rank_chunks(question,kp['chunks'],kp['summary'],min(6,settings.get('retrieval',{}).get('top_k',DEFAULT_TOP_K))); context=compose_context(kp['summary'],ranked,22000); candidate_blocks='\n\n'.join([f"## CANDIDATE {a.id}\n\n{a.response_text}" for a in candidates]); judge_user=f"Domanda originale:\n{question}\n\nCorpus rilevante:\n{context}\n\nRisposte candidate:\n{candidate_blocks}"; write_text(run_dir/'prompts'/'judge_prompt.txt',judge_user)
    try:
        if provider=='openrouter':
            api_key=os.getenv('OPENROUTER_API_KEY');
            if not api_key: raise RuntimeError('OPENROUTER_API_KEY non impostata')
            raw=call_openrouter(model,api_key,[{'role':'system','content':JUDGE_SYSTEM_PROMPT},{'role':'user','content':judge_user}],response_format={'type':'json_object'}); parsed=safe_json_loads(extract_openai_like_text(raw)) or {'raw_text':extract_openai_like_text(raw)}
        elif provider=='gemini':
            api_key=os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
            if not api_key: raise RuntimeError('GEMINI_API_KEY / GOOGLE_API_KEY non impostata')
            raw=call_gemini_generate_content(model,api_key,JUDGE_SYSTEM_PROMPT,judge_user); parsed=safe_json_loads(extract_gemini_text(raw)) or {'raw_text':extract_gemini_text(raw)}
        else: raise RuntimeError(f'Judge provider non supportato: {provider}')
        write_json(run_dir/'judge_result.json',parsed); return parsed
    except Exception as e:
        err={'error':str(e)}; write_json(run_dir/'judge_result.json',err); return err
def render_report(question:str,kp:Dict[str,Any],answers:List[Answer],judge:Optional[Dict[str,Any]],settings:Dict[str,Any])->str:
    ranked=rank_chunks(question,kp['chunks'],kp['summary'],settings.get('retrieval',{}).get('top_k',DEFAULT_TOP_K)); lines=['# Report multi-LLM avanzato','',f'**Domanda:** {question}','','## Knowledge pack',f"- documenti: {len(kp['documents'])}",f"- chunk: {len(kp['chunks'])}",'','## Summary tecnico',trim_text(kp['summary'].get('summary_text',json.dumps(kp['summary'],ensure_ascii=False,indent=2)),12000),'','## Chunk principali']
    for rc in ranked: lines.append(f"- `{rc.chunk.chunk_id}` | `{rc.chunk.source_path}` | score={rc.score:.2f} | headings={', '.join(rc.chunk.headings)}")
    lines+=['','## Risposte']
    for a in answers:
        lines += [f'### {a.id}',f'- success: {str(a.success).lower()}',f'- latency_s: {a.latency_s:.2f}']
        if a.error: lines.append(f'- error: {a.error}')
        lines.append('')
        if a.success: lines += [a.response_text,'']
    if judge: lines += ['## Judge',json.dumps(judge,ensure_ascii=False,indent=2),'']
    return '\n'.join(lines)
def save_outputs(run_dir:Path,question:str,kp:Dict[str,Any],answers:List[Answer],judge:Optional[Dict[str,Any]],settings:Dict[str,Any])->None:
    write_json(run_dir/'answers.json',[asdict(a) for a in answers])
    with (run_dir/'answers.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['id','provider','model','profile','success','latency_s','error','response_text','prompt_path']); w.writeheader()
        for a in answers: w.writerow({'id':a.id,'provider':a.provider,'model':a.model,'profile':a.profile,'success':a.success,'latency_s':f'{a.latency_s:.2f}','error':a.error or '','response_text':a.response_text,'prompt_path':a.prompt_path or ''})
    write_text(run_dir/'report.md',render_report(question,kp,answers,judge,settings))
def load_config(config_path:Path)->Dict[str,Any]: return json.loads(read_text(config_path))
def targets_from_config(cfg:Dict[str,Any])->List[Target]:
    out=[]
    for t in cfg.get('targets',[]):
        if not t.get('enabled',True): continue
        out.append(Target(t['provider'],t['model'],t.get('profile','systems_designer'),t.get('temperature',0.4),t.get('max_tokens'),t.get('enabled',True),t.get('label')))
    return out
def create_run_dirs(output_root:Path,label:str)->Tuple[Path,Path]:
    rid=f"{now_ts()}-{slugify(label)[:40]}"; run_dir=ensure_dir(output_root/'runs'/rid); cache_dir=ensure_dir(output_root/'cache'/rid); ensure_dir(run_dir/'prompts'); return run_dir,cache_dir
def source_label(args)->str:
    if args.repo: return Path(args.repo.rstrip('/')).name.replace('.git','') or 'repo'
    if args.files: return 'files-batch'
    if args.cache_dir: return Path(args.cache_dir).name
    return 'run'
def parse_args():
    p=argparse.ArgumentParser(description='Pipeline avanzata multi-LLM per regolamenti GdR.'); g=p.add_mutually_exclusive_group(required=False); g.add_argument('--repo'); g.add_argument('--files',nargs='+'); p.add_argument('--cache-dir'); p.add_argument('--config',required=True); p.add_argument('--question'); p.add_argument('--build-only',action='store_true'); p.add_argument('--output-root',default=str(DEFAULT_OUTPUT_ROOT)); return p.parse_args()
def main()->int:
    args=parse_args(); cfg=load_config(Path(args.config)); output_root=ensure_dir(Path(args.output_root)); run_dir,default_cache_dir=create_run_dirs(output_root,source_label(args)); log_path=run_dir/'run_log.jsonl'; append_jsonl(log_path,{'event':'start','ts':now_ts(),'version':APP_VERSION})
    if args.cache_dir:
        cache_dir=Path(args.cache_dir); kp=load_knowledge_pack(cache_dir); append_jsonl(log_path,{'event':'cache_loaded','cache_dir':str(cache_dir)})
    else:
        if not args.repo and not args.files: print('Errore: specifica --repo oppure --files oppure --cache-dir',file=sys.stderr); return 2
        docs=load_documents_from_repo(args.repo,cfg.get('input',{}).get('ignore_patterns',DEFAULT_IGNORE_PATTERNS)) if args.repo else load_documents_from_files(args.files or [])
        if not docs: print('Nessun documento Markdown trovato.',file=sys.stderr); return 2
        cache_dir=default_cache_dir; kp=build_knowledge_pack(docs,cache_dir,cfg); append_jsonl(log_path,{'event':'knowledge_built','documents':len(docs),'cache_dir':str(cache_dir)})
    write_json(run_dir/'knowledge_summary.json',kp.get('summary',{})); write_json(run_dir/'knowledge_manifest.json',kp.get('manifest',{})); write_json(run_dir/'effective_config.json',cfg)
    if args.build_only: append_jsonl(log_path,{'event':'build_only_done','cache_dir':str(cache_dir)}); print(str(cache_dir)); return 0
    question=args.question or cfg.get('run',{}).get('question')
    if not question: print('Errore: manca la question, passala con --question o nel config JSON.',file=sys.stderr); return 2
    targets=targets_from_config(cfg)
    if not targets: print('Errore: nessun target abilitato nel config.',file=sys.stderr); return 2
    append_jsonl(log_path,{'event':'targets_ready','count':len(targets),'targets':[asdict(t) for t in targets]})
    answers=[]; max_workers=cfg.get('run',{}).get('max_workers',DEFAULT_MAX_WORKERS)
    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs=[ex.submit(ask_target,t,question,kp,run_dir,cfg) for t in targets]
        for fut in cf.as_completed(futs):
            ans=fut.result(); answers.append(ans); append_jsonl(log_path,{'event':'answer_done','id':ans.id,'success':ans.success,'latency_s':ans.latency_s,'error':ans.error})
    answers.sort(key=lambda a:a.id); judge=run_judge(question,answers,kp,cfg,run_dir); save_outputs(run_dir,question,kp,answers,judge,cfg)
    final={'run_dir':str(run_dir),'cache_dir':str(cache_dir),'answers_ok':sum(1 for a in answers if a.success),'answers_fail':sum(1 for a in answers if not a.success),'judge_enabled':bool(cfg.get('judge',{}).get('enabled',False))}
    write_json(run_dir/'run_index.json',final); append_jsonl(log_path,{'event':'done',**final}); print(str(run_dir)); return 0
if __name__=='__main__': raise SystemExit(main())
