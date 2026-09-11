import json,os,subprocess,sys
from pathlib import Path
rows=[]
for audit,games,seed in ((True,2,670000000),(False,4,670100000)):
 for i in range(games):
  pair=[]
  for mode in (('base','fast') if i%2==0 else ('fast','base')):
   path=Path('/out')/f'{seed+i}-{mode}.json'
   cmd=[sys.executable,'/study/fast/examples/benchmark_value.py','--seed',str(seed+i),'--out',str(path)]
   if audit:cmd.append('--audit')
   subprocess.run(cmd,env={**os.environ,'PYTHONPATH':f'/study/{mode}/catanatron'},check=True)
   row={'mode':mode,**json.loads(path.read_text())};rows.append(row);pair.append(row)
  fields=['seed','actions','action_sha256','winner','points']
  if audit:fields+=['leaves','leaf_sha256','deadline_observations']
  assert all(pair[0][key]==pair[1][key] for key in fields), pair
  Path('/out/comparison.json').write_text(json.dumps({'complete':len(rows)==12,'equivalent':True,'rows':rows},indent=2)+'\n')
