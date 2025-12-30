# Helper: convert docs/combined_docs.html -> docs/NeutroHydro_docs.pdf
# Tries Chrome, Edge, or wkhtmltopdf if installed. Run from repository root:
#   .\scripts\convert_html_to_pdf.ps1

$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not $repoRoot) { $repoRoot = Get-Location }
$html = Resolve-Path (Join-Path $repoRoot 'docs\combined_docs.html') -ErrorAction SilentlyContinue
if (-not $html) {
    Write-Error "docs/combined_docs.html not found. Generate it first: pandoc docs/combined_docs.md -o docs/combined_docs.html"
    exit 1
}
$output = Join-Path $repoRoot 'docs\NeutroHydro_docs.pdf'

$chromeCandidates = @("$env:ProgramFiles\Google\Chrome\Application\chrome.exe", "$env:ProgramFiles(x86)\Google\Chrome\Application\chrome.exe")
$edgeCandidates = @("$env:ProgramFiles\Microsoft\Edge\Application\msedge.exe", "$env:ProgramFiles(x86)\Microsoft\Edge\Application\msedge.exe")
$wkCandidates = @("$env:ProgramFiles\wkhtmltopdf\bin\wkhtmltopdf.exe", "$env:ProgramFiles(x86)\wkhtmltopdf\bin\wkhtmltopdf.exe")

function Try-Run($exe, $argList) {
    Write-Host "Running: $exe with $argList"
    $proc = Start-Process -FilePath $exe -ArgumentList $argList -Wait -NoNewWindow -PassThru -ErrorAction SilentlyContinue
    return $proc -and $proc.ExitCode -eq 0
}

# Chrome
foreach ($c in $chromeCandidates) {
    if (Test-Path $c) {
        $argList = @('--headless', '--disable-gpu', "--print-to-pdf=$output", "file:///$($html.Path)")
        if (Try-Run $c $argList) { Write-Host "PDF written to $output"; exit 0 }
    }
}

# Edge
foreach ($c in $edgeCandidates) {
    if (Test-Path $c) {
        $argList = @('--headless', '--disable-gpu', "--print-to-pdf=$output", "file:///$($html.Path)")
        if (Try-Run $c $argList) { Write-Host "PDF written to $output"; exit 0 }
    }
}

# wkhtmltopdf
foreach ($c in $wkCandidates) {
    if (Test-Path $c) {
        $argList = @($html.Path, $output)
        if (Try-Run $c $argList) { Write-Host "PDF written to $output"; exit 0 }
    }
}

Write-Error "No supported rendering engine found (Chrome/Edge/wkhtmltopdf). Install one of them or install a TeX distribution (MiKTeX/TeX Live) and run: pandoc docs/combined_docs.md -o docs/NeutroHydro_docs.pdf --pdf-engine=pdflatex"
exit 2
