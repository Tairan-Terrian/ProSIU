"""Two fixed stochastic Qwen libraries complement the released library."""
import json
import torch
from prosiu.common import DEVICE,ASSETS,read_json,write_json
from .core import OUTPUT


@torch.inference_mode()
def generate_banks():
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
    from transformers import AutoModelForCausalLM,AutoTokenizer
    pending=[s for s in [1,2] if not (OUTPUT/'program_banks'/f'bank_{s}'/'programs.json').exists()]
    if not pending: return
    path='/media/yjz/htr/llm_model/Qwen2.5-3B-Instruct'
    tokenizer=AutoTokenizer.from_pretrained(path)
    model=AutoModelForCausalLM.from_pretrained(path,torch_dtype=torch.bfloat16).to(DEVICE).eval()
    names=['additive','interaction','novelty','balance','separation']
    schema={'type':'array','minItems':8,'maxItems':8,'items':{'type':'object',
        'required':['name','weights','rank'],'properties':{'name':{'type':'string'},
        'weights':{'type':'object','required':names,
            'properties':{k:{'enum':[-3.,-2.,-1.,-.5,0.,.5,1.,2.,3.]} for k in names}},
        'rank':{'enum':list(range(8))}}}}
    prompts=read_json(ASSETS/'program_generation.json')['responses']
    for seed in pending:
        torch.manual_seed(76200+seed)
        programs,responses=[],[]
        for source in prompts:
            prompt=source['prompt'].split('Existing proposals:')[0]+'Existing proposals: '+json.dumps(programs)
            tokens=tokenizer.apply_chat_template([{'role':'user','content':prompt}],add_generation_prompt=True,
                return_tensors='pt',return_dict=True).to(DEVICE)
            generated=model.generate(**tokens,max_new_tokens=2500,do_sample=True,temperature=.7,top_p=.9,
                pad_token_id=tokenizer.eos_token_id,
                prefix_allowed_tokens_fn=build_transformers_prefix_allowed_tokens_fn(tokenizer,JsonSchemaParser(schema)))
            response=tokenizer.decode(generated[0,tokens['input_ids'].shape[1]:],skip_special_tokens=True)
            responses.append({'prompt':prompt,'response':response})
            programs.extend([{'name':r['name'],'weights':[r['weights'][k] for k in names],'rank':r['rank']} for r in json.loads(response)])
        folder=OUTPUT/'program_banks'/f'bank_{seed}'
        write_json(folder/'programs.json',programs)
        write_json(folder/'generation.json',{'model':path,'seed':76200+seed,'temperature':.7,'top_p':.9,'responses':responses})
        print({'stage':'generate_bank','seed':seed,'count':len(programs)},flush=True)
    del model
    torch.cuda.empty_cache()
