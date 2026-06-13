# A/B context-efficiency harness for perfdigest.
# Runs the SAME diagnosis task through Claude Sonnet 4.6 twice — once with only
# the perfdigest MCP tools (ON), once reading the raw ncu details dump (OFF) —
# and prints the profiler-payload comparison. Requires: claude CLI, uv, and a
# profiled fixture at perfdigest/tests/fixtures/gafime.ncu-rep.
$ErrorActionPreference = "Stop"
$kit       = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)   # ...\MCP_developing_kit
$pkg       = Join-Path $kit "perfdigest"
$report    = Join-Path $pkg "tests\fixtures\gafime.ncu-rep"
$rawTxt    = Join-Path $kit "test_script\raw_details.txt"
$mcp       = Join-Path $pkg ".mcp.json"
$model     = "claude-sonnet-4-6"

if (-not (Test-Path $report)) { throw "fixture missing: $report (profile first)" }

# Export the realistic OFF-arm baseline (reading an existing report; no GPU).
ncu -i $report --page details 2>$null | Out-File -FilePath $rawTxt -Encoding utf8

$task = "find the kernel launch with the highest duration, then decide whether it is memory-bound or compute-bound and name the single biggest limiting factor, citing specific metric values. Answer in 3 sentences."

Write-Host "== ON arm (perfdigest MCP tools) =="
Push-Location $pkg
$promptA = "A CUDA kernel was profiled with NVIDIA Nsight Compute into the report at $report (format 'ncu-rep'). Using ONLY the perfdigest MCP tools, $task"
$a = $promptA | claude -p --model $model --output-format json --mcp-config $mcp --strict-mcp-config `
    --allowedTools mcp__perfdigest__list_kernels mcp__perfdigest__get_metrics mcp__perfdigest__expand | ConvertFrom-Json
Pop-Location

Write-Host "== OFF arm (raw Read) =="
$promptB = "A CUDA kernel was profiled with NVIDIA Nsight Compute. The full profiler output is in the text file at $rawTxt. Read it, $task"
$b = $promptB | claude -p --model $model --output-format json --allowedTools Read | ConvertFrom-Json

"`n--- RESULT ---"
"ON  turns=$($a.num_turns) cost=`$$([math]::Round($a.total_cost_usd,4))  answer: $($a.result.Substring(0,[Math]::Min(120,$a.result.Length)))..."
"OFF turns=$($b.num_turns) cost=`$$([math]::Round($b.total_cost_usd,4))  answer: $($b.result.Substring(0,[Math]::Min(120,$b.result.Length)))..."
"`nProfiler payload bytes:"
"  raw details : $((Get-Item $rawTxt).Length)"
"  digest      : ~1541 (list_kernels + get_metrics JSON)"
"See eval/RESULTS.md for the full write-up."
