Param(
  [ValidateSet("whoosh","tantivy")]
  [string]$Backend = "tantivy"
)
# Usa Python+PyYAML para actualización segura del nodo retrieval.lexical.backend
$py = @"
import sys, yaml
cfg = yaml.safe_load(open('config/universal.yaml','r',encoding='utf-8'))
cfg.setdefault('retrieval',{}).setdefault('lexical',{})['backend'] = sys.argv[1]
open('config/universal.yaml','w',encoding='utf-8').write(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
"@
python - <<PY $Backend
$py
PY
Write-Host "Lexical backend -> $Backend (config/universal.yaml)"
