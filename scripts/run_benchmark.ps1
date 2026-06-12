# Run AnRAG retrieval benchmark (no LLM / VLM / OCR)
# Usage:
#   .\scripts\run_benchmark.ps1
#   .\scripts\run_benchmark.ps1 -Corpus benchmarks\sample\corpus.txt -Qa benchmarks\sample\qa.jsonl
#   .\scripts\run_benchmark.ps1 -Corpus benchmarks\sample\hotpot_sample.jsonl -Format hotpotqa
#   .\scripts\run_benchmark.ps1 -Corpus path\to\ragbench.jsonl -Format ragbench

param(
    [string]$Corpus = "benchmarks\sample\corpus.txt",
    [string]$Qa = "benchmarks\sample\qa.jsonl",
    [string]$Format = "anrag",
    [int]$TopK = 8
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

Write-Host "Installing dependencies..."
python -m pip install -q -r requirements.txt
python -m pip install -q -e .

Write-Host "Running AnRAG benchmark..."
if ($Format -eq "anrag") {
    python -m anrag.cli ablation $Corpus --qa $Qa --format anrag --top-k $TopK
} else {
    python -m anrag.cli ablation $Corpus --format $Format --top-k $TopK
}
