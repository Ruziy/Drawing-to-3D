"""Small local-only HTTP client. Does not inherit HTTP proxies or follow redirects."""
from dataclasses import dataclass
import base64
import hashlib
import ipaddress
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

@dataclass(frozen=True)
class LLMSettings:
    enabled: bool = True
    base_url: str = 'http://127.0.0.1:11434'
    model: str = 'qwen3.5:9b'
    timeout: int = 180
    max_retries: int = 1
    analysis_mode: str = 'hybrid'
    max_iterations: int = 3

    @classmethod
    def from_env(cls, mode=None, model=None):
        def number(key, default, low, high):
            try: return max(low, min(high, int(os.environ.get(key, default))))
            except ValueError: return default
        return cls(enabled=os.environ.get('LLM_ENABLED','true').lower() in ('1','true','yes'),
                   base_url=os.environ.get('LLM_BASE_URL',cls.base_url).rstrip('/'),
                   model=model or os.environ.get('LLM_MODEL',cls.model),
                   timeout=number('LLM_TIMEOUT',180,5,600),
                   max_retries=number('LLM_MAX_RETRIES',1,0,2),
                   analysis_mode=mode or os.environ.get('LLM_ANALYSIS_MODE','hybrid'),
                   max_iterations=number('LLM_MAX_ITERATIONS',3,1,3))

    def validate_local(self):
        url = urllib.parse.urlsplit(self.base_url)
        host = (url.hostname or '').lower()
        try: local = ipaddress.ip_address(host).is_loopback
        except ValueError: local = host in ('localhost', 'host.docker.internal')
        if url.scheme != 'http' or not local or url.username or url.password or url.path not in ('','/') or url.query or url.fragment:
            raise ValueError('LLM_BASE_URL должен указывать на локальную Ollama: http://127.0.0.1:11434')
        validate_model_name(self.model)
        if self.analysis_mode not in ('classic','hybrid'):
            raise ValueError('LLM_ANALYSIS_MODE: classic или hybrid')


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError('Перенаправления Ollama запрещены: изображения должны оставаться локально')


def validate_model_name(model):
    # Names are identifiers in the local inventory, never URLs or CLI options.
    if (not isinstance(model,str) or len(model)>200 or
        not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*(?:/[A-Za-z0-9][A-Za-z0-9_.-]*)*(?::[A-Za-z0-9][A-Za-z0-9_.-]*)?',model)
        or 'cloud' in model.lower()):
        raise ValueError(f'Недопустимое имя локальной модели Ollama: {model!r}')


def normalize_json_content(content):
    """Accept one complete JSON object, optionally fenced; never guess fields.

    A prose wrapper is allowed only when it contains exactly one decodable
    object and no other JSON objects/arrays. The existing schema still decides
    whether that object is a valid CAD plan.
    """
    content=content.strip()
    def valid(raw):
        value=json.loads(raw,parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
        if not isinstance(value,dict):raise ValueError('Ожидается JSON-объект')
        return json.dumps(value,ensure_ascii=False,allow_nan=False)
    try:return valid(content)
    except (ValueError,TypeError):pass
    match=re.fullmatch(r'```(?:json)?\s*(.*?)\s*```',content,re.S|re.I)
    if match:return valid(match[1])
    decoder=json.JSONDecoder(parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    start=content.find('{')
    if start<0:raise ValueError('Ollama не вернула JSON-объект')
    value,end=decoder.raw_decode(content,start)
    if any(c in content[:start]+content[end:] for c in '{}[]'):
        raise ValueError('Неоднозначный ответ Ollama: несколько структур JSON')
    return valid(json.dumps(value,ensure_ascii=False,allow_nan=False))


class OllamaClient:
    def __init__(self, settings=None):
        self.settings = settings or LLMSettings.from_env()
        self.settings.validate_local()
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def request(self, endpoint, payload=None, timeout=None):
        data = json.dumps(payload,ensure_ascii=False,allow_nan=False).encode() if payload is not None else None
        request = urllib.request.Request(self.settings.base_url+endpoint, data,
                                         {'Content-Type':'application/json'})
        with self.opener.open(request, timeout=timeout or self.settings.timeout) as response:
            body = response.read(8*1024*1024+1)
            if len(body)>8*1024*1024: raise ValueError('Ответ Ollama превышает 8 МБ')
            return json.loads(body)

    def inventory(self):
        """One capability-based discovery mechanism for UI, API and worker."""
        version=self.request('/api/version',timeout=3)
        models=[]
        for tag in self.request('/api/tags',timeout=3).get('models',[]):
            name=tag.get('name',tag.get('model',''))
            row={'name':name,'digest':tag.get('digest'),'compatible':False,'capabilities':[]}
            try:
                validate_model_name(name)
                info=self.request('/api/show',{'model':name},timeout=5)
                row['capabilities']=info.get('capabilities',[])
                if any(info.get(k) or tag.get(k) for k in ('remote_host','remote_model')):
                    raise ValueError('Облачная привязка Ollama запрещена; используются только локальные веса')
                if 'vision' not in row['capabilities']:
                    raise ValueError('Модель не поддерживает изображения')
                row.update(compatible=True,reason='Локальная мультимодальная модель готова')
            except Exception as exc:row['reason']=str(exc)
            models.append(row)
        return {'version':version.get('version'),'models':models}

    def status(self):
        result = {'enabled':self.settings.enabled, 'model':self.settings.model,
                  'mode':self.settings.analysis_mode,'available':False,'local_only':True,
                  'base_url':self.settings.base_url,'installed_models':[]}
        if not self.settings.enabled:
            return {**result,'reason':'LLM отключена; доступен CV/OCR'}
        try:
            inventory=self.inventory()
            result.update(inventory,installed_models=[m['name'] for m in inventory['models'] if m['compatible']])
            selected=next((m for m in inventory['models'] if m['name']==self.settings.model),None)
            if selected is None:
                return {**result,'reason':f'Модель {self.settings.model} не установлена в локальной Ollama. Выполните ollama pull {self.settings.model}'}
            result.update(available=selected['compatible'],capabilities=selected['capabilities'],digest=selected['digest'],
                          reason=f"{self.settings.model}: {selected['reason']}")
        except Exception as exc:
            result['reason']=f'Ollama недоступна для модели {self.settings.model}: {exc}. Доступен режим CV/OCR.'
        return result

    def require_available(self):
        status=self.status()
        if not status['available']:raise ValueError(status['reason'])
        return status

    def chat(self, messages, schema):
        self.require_available()  # Recheck before sending any image, even after queue delay.
        payload = {'model':self.settings.model,'messages':messages,'format':schema,
                   'stream':True,'think':False,'keep_alive':'10m',
                   'options':{'temperature':0.2,'top_p':0.9,'seed':42,
                              'num_ctx':int(os.environ.get('LLM_NUM_CTX','16384')),
                              'num_predict':int(os.environ.get('LLM_NUM_PREDICT','5000'))}}
        # Only retry transport/temporary server failures. Invalid JSON is repaired
        # by the bounded semantic loop, not by an unbounded HTTP retry.
        for retry in range(self.settings.max_retries+1):
            try:
                response=self._bounded_chat(payload)
                actual=response.get('model')
                if actual != self.settings.model:
                    raise ValueError(f'Ollama: запрошена {self.settings.model}, получен ответ модели {actual!r}')
                response.update(requested_model=self.settings.model,actual_model=actual,ollama_status='responded')
                response['image_inputs']=[{'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
                    for message in messages for encoded in message.get('images',[])
                    for raw in [base64.b64decode(encoded,validate=True)]]
                raw=response.get('message',{}).get('content','')
                try:response['message']['content']=normalize_json_content(raw)
                except ValueError:pass  # Existing bounded schema repair gets the original response.
                else:
                    if response['message']['content']!=raw:response['raw_content']=raw
                return response
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                if isinstance(exc,urllib.error.HTTPError):
                    detail=exc.read(2000).decode('utf-8','replace')
                    # Preserve 5xx body too: it contains memory/model-load diagnostics.
                    if exc.code<500 or retry>=self.settings.max_retries:
                        raise ValueError(f'Ollama {self.settings.model}, HTTP {exc.code}: {detail}') from exc
                    continue
                if retry>=self.settings.max_retries:
                    raise RuntimeError(f'Ollama {self.settings.model}: {exc}') from exc

    def _bounded_chat(self, payload):
        """Some Qwen VL thinking builds ignore think=false. Close a real, bounded
        thinking stream and resume with that actual assistant trace, once only.
        No model weights, Ollama installation or user's other models are changed.
        """
        request=urllib.request.Request(self.settings.base_url+'/api/chat',
                                      json.dumps(payload,ensure_ascii=False,allow_nan=False).encode(),
                                      {'Content-Type':'application/json'})
        thinking=[];content=[];chunks=0;size=0;begun=time.monotonic();last={};limited=False
        with self.opener.open(request,timeout=self.settings.timeout) as response:
            for raw in response:
                size+=len(raw)
                if size>8*1024*1024:raise ValueError('Поток Ollama превышает 8 МБ')
                if time.monotonic()-begun>self.settings.timeout:raise TimeoutError('Превышен лимит времени Ollama')
                item=json.loads(raw)
                if item.get('error'):raise ValueError(item['error'])
                message=item.get('message',{})
                if message.get('thinking'):
                    thinking.append(message['thinking']);chunks+=1
                content.append(message.get('content',''));last=item
                if item.get('done'):break
                if chunks>=500 and not any(content):limited=True;break
        trace=''.join(thinking)
        if limited:
            remaining=self.settings.timeout-(time.monotonic()-begun)
            if remaining<5:raise TimeoutError('Не осталось времени на итоговый JSON Ollama')
            continuation={**payload,'stream':False,
                          'messages':payload['messages']+[{'role':'assistant','thinking':trace,'content':'\n'}]}
            final=self.request('/api/chat',continuation,timeout=remaining)
            final.update(reasoning_budget_applied=True,initial_thinking=trace,http_calls=2)
            return final
        last['message']={'role':'assistant','content':''.join(content),'thinking':trace}
        last.update(reasoning_budget_applied=False,http_calls=1)
        return last


def llm_status(settings=None):
    settings = settings or LLMSettings.from_env()
    try: return OllamaClient(settings).status()
    except ValueError as exc:
        return {'available':False,'enabled':settings.enabled,'model':settings.model,
                'mode':settings.analysis_mode,'local_only':True,'reason':str(exc),'installed_models':[]}
