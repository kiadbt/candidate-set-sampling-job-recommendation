param(
    [string]$ProjectRoot = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..\..")).Path
}

$datasetRoot = Join-Path $ProjectRoot "Dataset\CareerBuilder_InternalTemporal_v1\First7Last2"
$outputRoot = Join-Path (Split-Path $PSScriptRoot -Parent) "aggregate_results\revision_r1\careerbuilder_shared_catalog_feasibility"
$resultPath = Join-Path $outputRoot "shared_catalog_feasibility_by_window.csv"
$qaJsonPath = Join-Path $outputRoot "INDEPENDENT_QA.json"
$qaMarkdownPath = Join-Path $outputRoot "INDEPENDENT_QA.md"
$nominalCatalogSize = 5000

if (-not (Test-Path -LiteralPath $resultPath -PathType Leaf)) {
    throw "Missing primary result file: $resultPath"
}

$reportedRows = @(Import-Csv -LiteralPath $resultPath)
$failures = [System.Collections.Generic.List[string]]::new()
$verificationRows = [System.Collections.Generic.List[object]]::new()

if ($reportedRows.Count -ne 7) {
    $failures.Add("Expected 7 reported window rows; found $($reportedRows.Count).")
}

foreach ($window in 1..7) {
    $windowDir = Join-Path $datasetRoot ("window_{0}" -f $window)
    $audit = @(Import-Csv -LiteralPath (Join-Path $windowDir "cohort_audit_sample.csv"))
    $population = @(Import-Csv -LiteralPath (Join-Path $windowDir "cohort_population.csv"))
    $targets = @(Import-Csv -LiteralPath (Join-Path $windowDir "valid_future_targets.csv"))

    $auditLookup = @{}
    foreach ($row in $audit) {
        $auditLookup[[string]$row.UserID] = $true
    }

    $auditTargetRows = @($targets | Where-Object { $auditLookup.ContainsKey([string]$_.UserID) })
    $auditUniqueJobs = @($auditTargetRows.JobID | Sort-Object -Unique).Count
    $populationUniqueJobs = @($targets.JobID | Sort-Object -Unique).Count
    $reported = @($reportedRows | Where-Object { [int]$_.window -eq $window })

    if ($reported.Count -ne 1) {
        $failures.Add("Window $window has $($reported.Count) reported rows instead of 1.")
        continue
    }

    $expected = [ordered]@{
        audit_users = $audit.Count
        population_users = $population.Count
        audit_target_events = $auditTargetRows.Count
        population_target_events = $targets.Count
        audit_unique_target_jobs = $auditUniqueJobs
        population_unique_target_jobs = $populationUniqueJobs
        audit_target_share_of_5000 = [math]::Round($auditUniqueJobs / $nominalCatalogSize, 4)
        audit_non_target_capacity_in_5000 = [math]::Max(0, $nominalCatalogSize - $auditUniqueJobs)
        audit_shared_5000_target_complete = ($auditUniqueJobs -le $nominalCatalogSize)
        population_target_requirement_ratio_to_5000 = [math]::Round($populationUniqueJobs / $nominalCatalogSize, 4)
        population_excess_target_jobs_over_5000 = [math]::Max(0, $populationUniqueJobs - $nominalCatalogSize)
        population_shared_5000_target_complete = ($populationUniqueJobs -le $nominalCatalogSize)
    }

    foreach ($field in $expected.Keys) {
        $reportedValue = $reported[0].$field
        $expectedValue = $expected[$field]
        if ($expectedValue -is [bool]) {
            if ([bool]::Parse([string]$reportedValue) -ne $expectedValue) {
                $failures.Add("Window $window field ${field}: expected $expectedValue, reported $reportedValue.")
            }
        }
        elseif ($expectedValue -is [double]) {
            if ([math]::Abs(([double]$reportedValue) - $expectedValue) -gt 0.0000001) {
                $failures.Add("Window $window field ${field}: expected $expectedValue, reported $reportedValue.")
            }
        }
        else {
            if ([int64]$reportedValue -ne [int64]$expectedValue) {
                $failures.Add("Window $window field ${field}: expected $expectedValue, reported $reportedValue.")
            }
        }
    }

    $verificationRows.Add([pscustomobject]@{
        window = $window
        audit_unique_target_jobs = $auditUniqueJobs
        population_unique_target_jobs = $populationUniqueJobs
        audit_target_share_of_5000 = $expected.audit_target_share_of_5000
        audit_non_target_capacity_in_5000 = $expected.audit_non_target_capacity_in_5000
        population_excess_target_jobs_over_5000 = $expected.population_excess_target_jobs_over_5000
        row_matches = ($failures | Where-Object { $_ -like "Window $window *" }).Count -eq 0
    })
}

$pass = $failures.Count -eq 0
$qaRecord = [ordered]@{
    analysis = "Independent verification of shared 5,000-job catalog feasibility"
    generated_utc = [DateTime]::UtcNow.ToString("o")
    method = "Independent Import-Csv, hashtable membership, and Sort-Object -Unique calculations."
    reported_result_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $resultPath).Hash
    pass = $pass
    failures = $failures
    verified_rows = $verificationRows
}
$qaRecord | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $qaJsonPath -Encoding utf8

$status = if ($pass) { "PASS" } else { "FAIL" }
$qaMarkdown = @"
# Independent QA: Shared 5,000-Job Catalog Feasibility

**Status:** $status

The seven window-level results were recomputed independently from the prepared audit cohorts, full eligible populations, and valid future targets. The verifier uses hashtable membership and sorted unique JobID lists rather than the primary script's HashSet aggregation.

Checks covered audit and population user counts, target-event counts, target-union counts, target shares, remaining non-target capacity, excess target requirements, and target-complete feasibility.

**Failures:** $($failures.Count)

The machine-readable verification record is INDEPENDENT_QA.json.
"@
$qaMarkdown | Set-Content -LiteralPath $qaMarkdownPath -Encoding utf8

if (-not $pass) {
    throw "Independent verification failed. See $qaJsonPath"
}

Write-Output "PASS"
Write-Output $qaJsonPath
Write-Output $qaMarkdownPath
