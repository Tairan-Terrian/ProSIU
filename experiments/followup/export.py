"""Collect actual follow-up records for local plotting and manuscript tables."""
import json
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from .core import OUTPUT

records=[]
counts={}
for experiment in [f'E{i:02}' for i in range(1,13)]:
    files=sorted((OUTPUT/experiment).glob('*/*/*/result.json'))
    counts[experiment]=len(files)
    for path in files:
        record=json.loads(path.read_text())
        record['source_path']=str(path.relative_to(OUTPUT))
        records.append(record)
folder=OUTPUT/'exports'
folder.mkdir(parents=True,exist_ok=True)
inventory=[]
for path in sorted((OUTPUT/'external_data/processed').glob('*.json')):
    source=json.loads(path.read_text())
    inventory.append({'experiment':'E10','assay':source['assay'],'counts':source['counts'],
        'eligible_targets':source['eligible_targets'],'measurement_scale':source['measurement_scale'],
        'wt_reference_origin':source['wt_reference_origin']})
for path in sorted((OUTPUT/'extrapolation_data').glob('*/processed/*.json')):
    source=json.loads(path.read_text())
    inventory.append({'experiment':'E11','assay':source['assay'],'split':path.parents[1].name,
        'counts':source['counts']})
(folder/'all_results.json').write_text(json.dumps({'counts':counts,'inventory':inventory,'records':records},indent=2)+'\n')
(folder/'dataset_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
with ZipFile(folder/'measured_results.zip','w',ZIP_DEFLATED) as archive:
    archive.write(folder/'all_results.json','all_results.json')
    archive.write(folder/'dataset_inventory.json','dataset_inventory.json')
    for path in sorted(OUTPUT.glob('E*/*/*/*/result.json')):
        archive.write(path,str(path.relative_to(OUTPUT)))
    for path in sorted(OUTPUT.glob('E*/*/*/*/parameters.pt')):
        archive.write(path,str(path.relative_to(OUTPUT)))
    for path in sorted(OUTPUT.glob('E*/*/*/*/reference_parameters.pt')):
        archive.write(path,str(path.relative_to(OUTPUT)))
    for path in sorted((OUTPUT/'program_banks').rglob('*.json')):
        archive.write(path,str(path.relative_to(OUTPUT)))
    archive.write(Path(__file__).with_name('protocol.json'),'protocol.json')
    for path in sorted(Path(__file__).parent.glob('*.py')):
        archive.write(path,'experiment_source/followup/'+path.name)
    archive.write(Path(__file__).parents[1]/'run_main12.sh','experiment_source/run_main12.sh')
    for name in ['audit.json','likelihood_quadrature.json','protocol.txt']:
        for path in folder.glob(name):
            archive.write(path,name)
print(json.dumps(counts),flush=True)
