param(
    [string]$ProjectRoot = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..\..")).Path
}

$datasetRoot = Join-Path $ProjectRoot "Dataset\CareerBuilder_InternalTemporal_v1\First7Last2"
$outputRoot = Join-Path (Split-Path $PSScriptRoot -Parent) "aggregate_results\revision_r1\careerbuilder_shared_catalog_feasibility"
[System.IO.Directory]::CreateDirectory($outputRoot) | Out-Null
$outputCsv = Join-Path $outputRoot "shared_catalog_feasibility_by_window.csv"
$outputJson = Join-Path $outputRoot "shared_catalog_feasibility_audit.json"
$outputReadme = Join-Path $outputRoot "README.md"
$nominalCatalogSize = 5000

function New-StringSet {
    Write-Output -NoEnumerate ([System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::Ordinal))
}

function Add-SetValues {
    param(
        [System.Collections.Generic.HashSet[string]]$Set,
        [object[]]$Rows,
        [string]$Property
    )
    foreach ($row in $Rows) {
        [void]$Set.Add([string]$row.$Property)
    }
}

$rows = [System.Collections.Generic.List[object]]::new()
$sourceHashes = [System.Collections.Generic.List[object]]::new()
$checks = [ordered]@{
    seven_windows_present = $true
    audit_cohort_is_1000_per_window = $true
    audit_users_are_population_subset = $true
    target_users_are_population_subset = $true
    all_audit_users_have_targets = $true
    audit_shared_5000_feasible_all_windows = $true
    population_shared_5000_infeasible_all_windows = $true
}

foreach ($window in 1..7) {
    $windowDir = Join-Path $datasetRoot ("window_{0}" -f $window)
    if (-not (Test-Path -LiteralPath $windowDir -PathType Container)) {
        $checks.seven_windows_present = $false
        throw "Missing prepared window directory: $windowDir"
    }

    $auditPath = Join-Path $windowDir "cohort_audit_sample.csv"
    $populationPath = Join-Path $windowDir "cohort_population.csv"
    $targetPath = Join-Path $windowDir "valid_future_targets.csv"

    foreach ($path in @($auditPath, $populationPath, $targetPath)) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Missing prepared input: $path"
        }
        $sourceHashes.Add([pscustomobject]@{
            window = $window
            file = [System.IO.Path]::GetFileName($path)
            sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash
        })
    }

    $audit = @(Import-Csv -LiteralPath $auditPath)
    $population = @(Import-Csv -LiteralPath $populationPath)
    $targets = @(Import-Csv -LiteralPath $targetPath)

    $auditUsers = New-StringSet
    $populationUsers = New-StringSet
    Add-SetValues -Set $auditUsers -Rows $audit -Property "UserID"
    Add-SetValues -Set $populationUsers -Rows $population -Property "UserID"

    if ($auditUsers.Count -ne 1000) {
        $checks.audit_cohort_is_1000_per_window = $false
    }
    foreach ($user in $auditUsers) {
        if (-not $populationUsers.Contains($user)) {
            $checks.audit_users_are_population_subset = $false
        }
    }

    $auditTargetJobs = New-StringSet
    $populationTargetJobs = New-StringSet
    $auditTargetUsers = New-StringSet
    $targetUsers = New-StringSet
    $auditTargetEvents = 0

    foreach ($target in $targets) {
        $user = [string]$target.UserID
        $job = [string]$target.JobID
        [void]$targetUsers.Add($user)
        [void]$populationTargetJobs.Add($job)
        if (-not $populationUsers.Contains($user)) {
            $checks.target_users_are_population_subset = $false
        }
        if ($auditUsers.Contains($user)) {
            $auditTargetEvents++
            [void]$auditTargetUsers.Add($user)
            [void]$auditTargetJobs.Add($job)
        }
    }

    if ($auditTargetUsers.Count -ne $auditUsers.Count) {
        $checks.all_audit_users_have_targets = $false
    }

    $auditFeasible = $auditTargetJobs.Count -le $nominalCatalogSize
    $populationFeasible = $populationTargetJobs.Count -le $nominalCatalogSize
    if (-not $auditFeasible) {
        $checks.audit_shared_5000_feasible_all_windows = $false
    }
    if ($populationFeasible) {
        $checks.population_shared_5000_infeasible_all_windows = $false
    }

    $rows.Add([pscustomobject]@{
        window = $window
        nominal_shared_catalog_size = $nominalCatalogSize
        audit_users = $auditUsers.Count
        population_users = $populationUsers.Count
        audit_target_events = $auditTargetEvents
        population_target_events = $targets.Count
        audit_unique_target_jobs = $auditTargetJobs.Count
        population_unique_target_jobs = $populationTargetJobs.Count
        audit_target_share_of_5000 = [math]::Round($auditTargetJobs.Count / $nominalCatalogSize, 4)
        audit_non_target_capacity_in_5000 = [math]::Max(0, $nominalCatalogSize - $auditTargetJobs.Count)
        audit_shared_5000_target_complete = $auditFeasible
        population_target_requirement_ratio_to_5000 = [math]::Round($populationTargetJobs.Count / $nominalCatalogSize, 4)
        population_excess_target_jobs_over_5000 = [math]::Max(0, $populationTargetJobs.Count - $nominalCatalogSize)
        population_shared_5000_target_complete = $populationFeasible
    })
}

$rows | Export-Csv -LiteralPath $outputCsv -NoTypeInformation -Encoding utf8

$auditShares = @($rows | ForEach-Object { [double]$_.audit_target_share_of_5000 })
$auditCapacities = @($rows | ForEach-Object { [int]$_.audit_non_target_capacity_in_5000 })
$auditUnions = @($rows | ForEach-Object { [int]$_.audit_unique_target_jobs })
$populationUnions = @($rows | ForEach-Object { [int]$_.population_unique_target_jobs })

$auditRecord = [ordered]@{
    analysis = "CareerBuilder shared 5,000-job target-completeness feasibility"
    generated_utc = [DateTime]::UtcNow.ToString("o")
    project_root = $ProjectRoot
    prepared_dataset = $datasetRoot
    nominal_shared_catalog_size = $nominalCatalogSize
    design = [ordered]@{
        audit = "Unique observed future target JobIDs among the 1,000-user audit cohort in each provider-defined window."
        population = "Unique observed future target JobIDs among the full eligible population in each provider-defined window."
        interpretation = "A target-complete shared catalog must contain the union of all observed target JobIDs for every included user."
    }
    aggregate = [ordered]@{
        audit_unique_target_jobs_min = ($auditUnions | Measure-Object -Minimum).Minimum
        audit_unique_target_jobs_max = ($auditUnions | Measure-Object -Maximum).Maximum
        audit_target_share_of_5000_min = [math]::Round(($auditShares | Measure-Object -Minimum).Minimum, 4)
        audit_target_share_of_5000_max = [math]::Round(($auditShares | Measure-Object -Maximum).Maximum, 4)
        audit_non_target_capacity_min = ($auditCapacities | Measure-Object -Minimum).Minimum
        audit_non_target_capacity_max = ($auditCapacities | Measure-Object -Maximum).Maximum
        population_unique_target_jobs_min = ($populationUnions | Measure-Object -Minimum).Minimum
        population_unique_target_jobs_max = ($populationUnions | Measure-Object -Maximum).Maximum
    }
    checks = $checks
    source_hashes = $sourceHashes
    output_csv_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $outputCsv).Hash
    rows = $rows
}

$auditRecord | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $outputJson -Encoding utf8

$auditMin = ($auditUnions | Measure-Object -Minimum).Minimum
$auditMax = ($auditUnions | Measure-Object -Maximum).Maximum
$shareMin = [math]::Round(100 * ($auditShares | Measure-Object -Minimum).Minimum, 2)
$shareMax = [math]::Round(100 * ($auditShares | Measure-Object -Maximum).Maximum, 2)
$capacityMin = ($auditCapacities | Measure-Object -Minimum).Minimum
$capacityMax = ($auditCapacities | Measure-Object -Maximum).Maximum
$populationMin = ($populationUnions | Measure-Object -Minimum).Minimum
$populationMax = ($populationUnions | Measure-Object -Maximum).Maximum
$auditMinText = ([int]$auditMin).ToString("N0")
$auditMaxText = ([int]$auditMax).ToString("N0")
$capacityMinText = ([int]$capacityMin).ToString("N0")
$capacityMaxText = ([int]$capacityMax).ToString("N0")
$populationMinText = ([int]$populationMin).ToString("N0")
$populationMaxText = ([int]$populationMax).ToString("N0")
$shareMinText = ([double]$shareMin).ToString("N2")
$shareMaxText = ([double]$shareMax).ToString("N2")

$readme = @"
# Shared 5,000-Job Catalog Feasibility

## Question

Could a single target-complete 5,000-job CareerBuilder catalog be shared by all evaluated users in a window without changing the cohort or dropping observed relevant jobs?

## Method

For each of the seven provider-defined windows, the analysis counts the union of unique observed future target JobIDs for (1) the 1,000-user audit cohort and (2) the full eligible population. A shared target-complete catalog must contain every job in that union. The calculation uses only the locked prepared cohort and target files; it does not alter sampling, models, scores, or metrics.

## Verified Findings

- In the 1,000-user audit cohorts, the target union ranges from $auditMinText to $auditMaxText jobs, occupying $shareMinText% to $shareMaxText% of a nominal 5,000-job shared catalog. Only $capacityMinText to $capacityMaxText positions would remain for non-target jobs.
- In the full eligible populations, the target union ranges from $populationMinText to $populationMaxText jobs. It exceeds 5,000 in every window, so a target-complete shared 5,000-job catalog is impossible without dropping users or observed targets.
- A shared audit-cohort catalog would be technically possible but target-dominated and would allocate a large, window-specific fraction of its capacity to the cohort target union. This changes the negative pool and makes the design less comparable across windows.
- The retained per-user, seed-matched nested design preserves every evaluated user and observed target while changing only local candidate scale. It is therefore the cleaner controlled scale sensitivity for the present estimand.

## Interpretation Boundary

This calculation supports the design choice; it does not establish that 5,000 candidates are universally sufficient or that per-user sampling reconstructs latent exposure or preference.

## Outputs

- shared_catalog_feasibility_by_window.csv: window-level counts and feasibility quantities.
- shared_catalog_feasibility_audit.json: design, integrity checks, source hashes, output hash, and row records.
- analyze_shared_catalog_feasibility.ps1: reproducible analysis script.
"@

$readme | Set-Content -LiteralPath $outputReadme -Encoding utf8

if ($checks.Values -contains $false) {
    throw "One or more fail-closed checks did not pass. See $outputJson"
}

Write-Output "PASS"
Write-Output $outputCsv
Write-Output $outputJson
Write-Output $outputReadme
