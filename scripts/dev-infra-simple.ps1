# RustRAG Development Infrastructure Management Script
# Usage: .\scripts\dev-infra-simple.ps1 [start|stop|status|logs|reset|health]

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("start", "stop", "status", "logs", "reset", "health")]
    [string]$Action
)

function Show-Status {
    Write-Host "Checking infrastructure status..." -ForegroundColor Cyan
    docker-compose ps
    Write-Host ""
}

function Start-Infrastructure {
    Write-Host "Starting RustRAG development infrastructure..." -ForegroundColor Green
    docker-compose up -d
    Write-Host ""
    Start-Sleep -Seconds 5
    Show-HealthChecks
}

function Stop-Infrastructure {
    Write-Host "Stopping RustRAG infrastructure..." -ForegroundColor Yellow
    docker-compose down
    Write-Host "Infrastructure stopped." -ForegroundColor Green
}

function Show-Logs {
    Write-Host "Showing infrastructure logs..." -ForegroundColor Cyan
    docker-compose logs -f
}

function Reset-Infrastructure {
    Write-Host "Resetting infrastructure (this will delete all data)..." -ForegroundColor Red
    $confirm = Read-Host "Are you sure? This will delete all data (y/N)"
    if ($confirm -eq "y" -or $confirm -eq "Y") {
        docker-compose down -v
        docker-compose up -d
        Write-Host "Infrastructure reset complete." -ForegroundColor Green
    } else {
        Write-Host "Reset cancelled." -ForegroundColor Yellow
    }
}

function Show-HealthChecks {
    Write-Host "Checking service health..." -ForegroundColor Cyan
    
    # Check PostgreSQL
    Write-Host "PostgreSQL (port 5433): " -NoNewline
    try {
        $result = docker exec rustrag-postgres pg_isready -U rustrag_user -d rustrag 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "OK - Healthy" -ForegroundColor Green
        } else {
            Write-Host "ERROR - Unhealthy" -ForegroundColor Red
        }
    } catch {
        Write-Host "ERROR - Connection failed" -ForegroundColor Red
    }
    
    # Check Qdrant
    Write-Host "Qdrant (port 6333): " -NoNewline
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:6333/" -Method GET -TimeoutSec 3 -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            Write-Host "OK - Healthy" -ForegroundColor Green
        } else {
            Write-Host "ERROR - Unhealthy" -ForegroundColor Red
        }
    } catch {
        Write-Host "ERROR - Unreachable" -ForegroundColor Red
    }
    
    # Check Redis
    Write-Host "Redis (port 6379): " -NoNewline
    try {
        $result = docker exec rustrag-redis redis-cli ping 2>$null
        if ($result -eq "PONG") {
            Write-Host "OK - Healthy" -ForegroundColor Green
        } else {
            Write-Host "ERROR - Unhealthy" -ForegroundColor Red
        }
    } catch {
        Write-Host "ERROR - Connection failed" -ForegroundColor Red
    }
    
    Write-Host ""
    Write-Host "Service URLs:" -ForegroundColor Cyan
    Write-Host "  PostgreSQL: postgresql://rustrag_user:rustrag_password@localhost:5433/rustrag"
    Write-Host "  Qdrant API: http://localhost:6333"
    Write-Host "  Redis: redis://localhost:6379"
    Write-Host ""
}

# Main script logic
switch ($Action) {
    "start" { Start-Infrastructure }
    "stop" { Stop-Infrastructure }
    "status" { Show-Status }
    "logs" { Show-Logs }
    "reset" { Reset-Infrastructure }
    "health" { Show-HealthChecks }
}
