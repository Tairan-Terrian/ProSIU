"""Identity-defined external cohorts and extrapolation partitions."""
from collections import defaultdict
import copy
import json
from pathlib import Path
import random
import shutil

import pandas as pd
import torch

from prosiu.common import write_json,read_json
from prosiu.data import sequence_of
from prosiu.features import build_context_features
from .core import DATA,RESEARCH,OUTPUT

EXTERNAL={
    'GB1':'SPG1_STRSG_Olson_2014',
    'YNZC':'YNZC_BACSU_Tsuboyama_2023_2JVD',
    'MYO3':'MYO3_YEAST_Tsuboyama_2023_2BTT',
}
EXTERNAL_CONFIG={
    'classification':{'features':'context','ridge':.001,'classification':1.,'preference':1.,'threshold':0.},
    'preference':{'features':'context','ridge':.001,'classification':0.,'preference':1.}}


def prepare_external():
    folder=OUTPUT/'external_data'
    inventory=[]
    for name,identifier in EXTERNAL.items():
        frame=pd.read_csv(RESEARCH/'data/raw'/(identifier+'.csv'))
        ref=list(frame.iloc[0].mutated_sequence)
        for m in frame.iloc[0].mutant.split(':'): ref[int(m[1:-1])-1]=m[0]
        wt=''.join(ref)
        def genotype(m): return tuple(sorted((int(x[1:-1])-1,x[-1]) for x in m.split(':')))
        measurements={genotype(r.mutant):float(r.DMS_score) for r in frame.itertuples()}
        # Both assay families report WT-relative measurements: lnW and signed ddG.
        measurements[()]=0.
        targets=[g for g in sorted(measurements) if len(g)==2 and (g[0],) in measurements and (g[1],) in measurements]
        order=list(range(len(targets)))
        random.Random(650017).shuffle(order)
        a,b,c=int(.4*len(order)),int(.6*len(order)),int(.8*len(order))
        parts={'train':order[:a][:4096],'dev_train':order[a:b][:2048],'dev':order[b:c][:1024],'test':order[c:][:2048]}
        rows=[]
        for part,indices in parts.items():
            for index in indices:
                g=targets[index]
                four=[(),(g[0],),(g[1],),g]
                rows.append({'q':[sequence_of(v,wt) for v in four],'y':[measurements[v] for v in four],
                    'positions':[v[0] for v in g],'depth':2,'partition':part})
        anchors=[{'sequence':sequence_of(g,wt),'y':measurements[g]} for g in sorted(measurements) if len(g)<=1]
        source={'assay':name,'identifier':identifier,'wt':wt,'rows':rows,'anchors':anchors,
            'anchor_ids':list(range(len(anchors))),'eligible_targets':len(targets),'split_seed':650017,
            'counts':{p:len(v) for p,v in parts.items()},'wt_reference_origin':0.,
            'measurement_scale':'WT-relative lnW' if name=='GB1' else 'WT-relative signed ddG',
            'eligibility':'measured double and both constituent singles; shared WT-relative assay origin'}
        write_json(folder/'processed'/f'{name}.json',source)
        inventory.append({k:source[k] for k in ['assay','identifier','eligible_targets','counts','measurement_scale']})
    write_json(folder/'inventory.json',inventory)
    torch.set_default_dtype(torch.float32)
    build_context_features(folder,list(EXTERNAL))
    torch.set_default_dtype(torch.float64)
    return folder


def split_source(source,kind):
    result=copy.deepcopy(source)
    rows=result['rows']
    if kind=='Random': return result
    if kind=='SeenConstituents':
        known={(p,r['q'][3][p]) for r in rows if r['partition']=='train' for p in r['positions']}
        for r in rows:
            if r['partition']=='test' and any((p,r['q'][3][p]) not in known for p in r['positions']): r['partition']='unused'
    else:
        randomizer=random.Random(77400)
        eligible=[i for i,r in enumerate(rows) if r['partition'] in ['train','test']]
        if kind=='PositionPair':
            groups=defaultdict(list)
            for i in eligible: groups[tuple(rows[i]['positions'])].append(i)
            keys=sorted(groups)
            randomizer.shuffle(keys)
            n=len(keys)
            cut=int(2*n/3)
            parts={'train':[i for key in keys[:cut] for i in groups[key]],
                   'test':[i for key in keys[cut:] for i in groups[key]]}
        else:
            identities=sorted({(p,rows[i]['q'][3][p]) for i in eligible for p in rows[i]['positions']})
            randomizer.shuffle(identities)
            unseen=set(identities[:int(.2*len(identities))])
            test=[i for i in eligible if any((p,rows[i]['q'][3][p]) in unseen for p in rows[i]['positions'])]
            observed=[i for i in eligible if all((p,rows[i]['q'][3][p]) not in unseen for p in rows[i]['positions'])]
            randomizer.shuffle(test)
            randomizer.shuffle(observed)
            parts={'train':observed,'test':test}
        for i in eligible: rows[i]['partition']='unused'
        caps={'train':4096,'test':2048}
        for part,ids in parts.items():
            for i in ids[:caps[part]]: rows[i]['partition']=part
    result['counts']={p:sum(r['partition']==p for r in rows) for p in ['train','dev_train','dev','test','unused']}
    result['followup_split']=kind
    result['development_identity_policy']='original dev_train/dev partitions preserved'
    return result


def prepare_splits(assays):
    torch.set_default_dtype(torch.float32)
    folders={}
    for kind in ['Random','PositionPair','SeenConstituents','UnseenConstituent']:
        folder=OUTPUT/'extrapolation_data'/kind
        folders[kind]=folder
        for assay in assays:
            source=split_source(read_json(DATA/'processed'/f'{assay}.json'),kind)
            write_json(folder/'processed'/f'{assay}.json',source)
        build_context_features(folder,assays)
        if 'amacGFP' in assays:
            shutil.copyfile(DATA/'features/amacGFP_language.pt',folder/'features/amacGFP_language.pt')
    torch.set_default_dtype(torch.float64)
    return folders
