
Param(
  [string]$Data = ".\data\eval\dev.jsonl",
  [int]$K = 6,
  [string]$Mode = "hybrid"
)
python .\tools\eval\eval_rag.py --data $Data --k $K --mode $Mode
