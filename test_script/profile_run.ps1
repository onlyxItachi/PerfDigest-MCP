Set-Location "C:\Users\hamza\Desktop\MCP_developing_kit\test_script"
& ncu --set full --force-overwrite -o "C:\Users\hamza\Desktop\MCP_developing_kit\perfdigest\tests\fixtures\gafime" .\gafime_bench.exe *> "C:\Users\hamza\Desktop\MCP_developing_kit\test_script\ncu_elevated.log"
"EXIT=$LASTEXITCODE" | Add-Content "C:\Users\hamza\Desktop\MCP_developing_kit\test_script\ncu_elevated.log"
