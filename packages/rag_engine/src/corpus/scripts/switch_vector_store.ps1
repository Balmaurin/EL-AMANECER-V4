
Param(
  [ValidateSet("local","qdrant","milvus")]
  [string]$Type = "local"
)
$py = @"
import sys, yaml
cfg = yaml.safe_load(open('config/universal.yaml','r',encoding='utf-8'))
cfg.setdefault('external_index',{})['type'] = sys.argv[1]
open('config/universal.yaml','w',encoding='utf-8').write(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
"@
python - <<PY $Type
$py
PY
Write-Host "Vector store -> $Type (config/universal.yaml)"
