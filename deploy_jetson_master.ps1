param(
    [string]$JetsonIP = "10.30.91.43",
    [string]$JetsonUser = "depilacjapl",
    [string]$JetsonPassword = "dupa1234",
    [string]$DeploymentMode = ""
)

Write-Host "=== JETSON NANO MASTER DEPLOYMENT ===" -ForegroundColor Green
Write-Host "Target: $JetsonUser@$JetsonIP" -ForegroundColor Yellow

# Interactive mode selection
if ($DeploymentMode -eq "") {
    Write-Host "`nChoose deployment mode:" -ForegroundColor Cyan
    Write-Host "  [venv]   - Python virtual environment (reliable, tested)" -ForegroundColor Green
    Write-Host "  [docker] - Docker containerized service (build remotely)" -ForegroundColor Yellow
    Write-Host "  [docker-local] - Docker service (build locally, deploy remotely)" -ForegroundColor Cyan
    Write-Host "  [upgrade] - Upgrade system Python to 3.10 first, then choose" -ForegroundColor Magenta
    
    do {
        $DeploymentMode = Read-Host "`nEnter your choice (venv/docker/docker-local/upgrade)"
        $DeploymentMode = $DeploymentMode.ToLower().Trim()
    } while ($DeploymentMode -notin @("venv", "docker", "docker-local", "upgrade"))
}

Write-Host "`nSelected mode: $DeploymentMode" -ForegroundColor Cyan

# Test SSH
ssh -o ConnectTimeout=5 "$JetsonUser@$JetsonIP" "echo 'SSH OK'" | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "SSH FAILED!" -ForegroundColor Red
    exit 1
}
Write-Host "SSH: OK" -ForegroundColor Green

# Handle Python upgrade if requested
if ($DeploymentMode -eq "upgrade") {
    Write-Host "`n=== UPGRADING SYSTEM PYTHON TO 3.10 ===" -ForegroundColor Magenta
    
    Write-Host "Adding deadsnakes PPA for Python 3.10..." -ForegroundColor Yellow
    ssh "$JetsonUser@$JetsonIP" "echo '$JetsonPassword' | sudo -S apt-get update -qq"
    ssh "$JetsonUser@$JetsonIP" "echo '$JetsonPassword' | sudo -S apt-get install -y software-properties-common"
    ssh "$JetsonUser@$JetsonIP" "echo '$JetsonPassword' | sudo -S add-apt-repository -y ppa:deadsnakes/ppa"
    ssh "$JetsonUser@$JetsonIP" "echo '$JetsonPassword' | sudo -S apt-get update -qq"
    
    Write-Host "Installing Python 3.10..." -ForegroundColor Yellow
    ssh "$JetsonUser@$JetsonIP" "echo '$JetsonPassword' | sudo -S apt-get install -y python3.10 python3.10-venv python3.10-dev python3.10-distutils"
    
    Write-Host "Setting Python 3.10 as default..." -ForegroundColor Yellow
    ssh "$JetsonUser@$JetsonIP" "echo '$JetsonPassword' | sudo -S update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1"
    ssh "$JetsonUser@$JetsonIP" "echo '$JetsonPassword' | sudo -S update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 2"
    ssh "$JetsonUser@$JetsonIP" "echo '$JetsonPassword' | sudo -S update-alternatives --set python3 /usr/bin/python3.10"
    
    # Install pip for Python 3.10
    Write-Host "Installing pip for Python 3.10..." -ForegroundColor Yellow
    ssh "$JetsonUser@$JetsonIP" "curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10"
    
    $pythonVersion = ssh "$JetsonUser@$JetsonIP" "python3 --version"
    Write-Host "Python upgrade complete: $pythonVersion" -ForegroundColor Green
    
    # Ask for deployment mode after upgrade
    Write-Host "`nNow choose deployment mode:" -ForegroundColor Cyan
    Write-Host "  [venv]   - Python 3.10 virtual environment" -ForegroundColor Green
    Write-Host "  [docker] - Docker containerized service" -ForegroundColor Yellow
    
    do {
        $DeploymentMode = Read-Host "`nEnter your choice (venv/docker)"
        $DeploymentMode = $DeploymentMode.ToLower().Trim()
    } while ($DeploymentMode -notin @("venv", "docker"))
    
    Write-Host "Selected mode after upgrade: $DeploymentMode" -ForegroundColor Cyan
}

# STEP 1: Transfer files (conditional based on mode)
if ($DeploymentMode -eq "docker-local") {
    Write-Host "`n=== STEP 1: PREPARING FOR LOCAL DOCKER BUILD ===" -ForegroundColor Cyan
    Write-Host "Creating directory on Jetson..." -ForegroundColor Yellow
    ssh "$JetsonUser@$JetsonIP" "mkdir -p ~/hairskin_classifier"
    Write-Host "Jetson directory prepared - will transfer Docker Compose after image build" -ForegroundColor Green
} else {
    Write-Host "`n=== STEP 1: TRANSFERRING PROJECT FILES ===" -ForegroundColor Cyan
    ssh "$JetsonUser@$JetsonIP" "mkdir -p ~/hairskin_classifier"

    Write-Host "Transferring code files..." -ForegroundColor Yellow
    scp "ml_service.py" "$JetsonUser@${JetsonIP}:~/hairskin_classifier/" 2>$null
    scp "ml_client.py" "$JetsonUser@${JetsonIP}:~/hairskin_classifier/" 2>$null
    scp "DepiModels.py" "$JetsonUser@${JetsonIP}:~/hairskin_classifier/" 2>$null
    scp "requirements.txt" "$JetsonUser@${JetsonIP}:~/hairskin_classifier/" 2>$null
    scp "start_ml_service.sh" "$JetsonUser@${JetsonIP}:~/hairskin_classifier/" 2>$null
    scp "docker-compose.yml" "$JetsonUser@${JetsonIP}:~/hairskin_classifier/" 2>$null
    scp "Dockerfile" "$JetsonUser@${JetsonIP}:~/hairskin_classifier/" 2>$null
}

# Transfer models and DICOM (skip for docker-local since they're in the image)
if ($DeploymentMode -ne "docker-local") {
    # Check for models
    $modelsExist = ssh "$JetsonUser@$JetsonIP" "test -d ~/hairskin_classifier/hc_model && test -d ~/hairskin_classifier/skin_type_model && echo 'YES' || echo 'NO'"
    if ($modelsExist -eq "NO") {
        Write-Host "Transferring models..." -ForegroundColor Yellow
        scp -r "hc_model" "$JetsonUser@${JetsonIP}:~/hairskin_classifier/" 2>$null
        scp -r "skin_type_model" "$JetsonUser@${JetsonIP}:~/hairskin_classifier/" 2>$null
        Write-Host "Models transferred" -ForegroundColor Green
    } else {
        Write-Host "Models exist - SKIPPED" -ForegroundColor Green
    }

    # Transfer test DICOM
    $dicom = Get-ChildItem "*.dcm" | Select-Object -First 1
    if ($dicom) {
        $dicomExists = ssh "$JetsonUser@$JetsonIP" "test -f ~/hairskin_classifier/$($dicom.Name) && echo 'YES' || echo 'NO'"
        if ($dicomExists -eq "NO") {
            Write-Host "Transferring test DICOM..." -ForegroundColor Yellow
            scp $dicom.FullName "$JetsonUser@${JetsonIP}:~/hairskin_classifier/" 2>$null
        }
    }
} else {
    Write-Host "Models and DICOM will be included in Docker image - SKIPPED" -ForegroundColor Green
}

# Make scripts executable (skip for docker-local)
if ($DeploymentMode -ne "docker-local") {
    ssh "$JetsonUser@$JetsonIP" "chmod +x ~/hairskin_classifier/start_ml_service.sh"
}

Write-Host "File preparation: DONE" -ForegroundColor Green

# STEP 2: Setup Python venv (only for venv mode)
if ($DeploymentMode -eq "venv") {
    $currentPython = ssh "$JetsonUser@$JetsonIP" "python3 --version"
    Write-Host "`n=== STEP 2: SETTING UP PYTHON VENV ===" -ForegroundColor Cyan
    Write-Host "Using Python: $currentPython" -ForegroundColor Green

    Write-Host "Creating venv in project folder..." -ForegroundColor Yellow
    $venvExists = ssh "$JetsonUser@$JetsonIP" "test -d ~/hairskin_classifier/venv && echo 'YES' || echo 'NO'"
    if ($venvExists -eq "NO") {
        ssh "$JetsonUser@$JetsonIP" "cd ~/hairskin_classifier && python3 -m venv venv"
        Write-Host "Virtual environment created" -ForegroundColor Green
    } else {
        Write-Host "Virtual environment exists - SKIPPED" -ForegroundColor Green
    }

    Write-Host "Upgrading pip..." -ForegroundColor Yellow
    ssh "$JetsonUser@$JetsonIP" "cd ~/hairskin_classifier && source venv/bin/activate && python -m pip install --upgrade pip"

    Write-Host "Installing requirements..." -ForegroundColor Yellow
    ssh "$JetsonUser@$JetsonIP" "cd ~/hairskin_classifier && source venv/bin/activate && pip install -r requirements.txt"

    Write-Host "Python venv: READY" -ForegroundColor Green
}

# STEP 3: Setup Docker (conditional based on mode)
if ($DeploymentMode -eq "docker" -or $DeploymentMode -eq "docker-local") {
    Write-Host "`n=== STEP 3: SETTING UP DOCKER ===" -ForegroundColor Cyan

    Write-Host "Starting Docker daemon..." -ForegroundColor Yellow
    ssh "$JetsonUser@$JetsonIP" "echo '$JetsonPassword' | sudo -S systemctl start docker 2>/dev/null || true"
    ssh "$JetsonUser@$JetsonIP" "echo '$JetsonPassword' | sudo -S usermod -aG docker $JetsonUser 2>/dev/null || true"

    # Test Docker
    $dockerTest = ssh "$JetsonUser@$JetsonIP" "docker info >/dev/null 2>&1 && echo 'OK' || echo 'SUDO'"
    if ($dockerTest -eq "SUDO") {
        Write-Host "Docker needs sudo - will use sudo for docker commands" -ForegroundColor Yellow
        $useSudo = $true
    } else {
        Write-Host "Docker: OK" -ForegroundColor Green
        $useSudo = $false
    }
} else {
    Write-Host "`n=== STEP 3: SKIPPING DOCKER SETUP ===" -ForegroundColor Yellow
    Write-Host "Using venv deployment mode" -ForegroundColor Green
    $useSudo = $false
}

# STEP 4: Deploy service based on selected mode
if ($DeploymentMode -eq "docker" -or $DeploymentMode -eq "docker-local") {
    if ($DeploymentMode -eq "docker-local") {
        Write-Host "`n=== STEP 4: BUILDING DOCKER SERVICE LOCALLY ===" -ForegroundColor Cyan
    } else {
        Write-Host "`n=== STEP 4: BUILDING DOCKER SERVICE REMOTELY ===" -ForegroundColor Cyan
    }
} else {
    Write-Host "`n=== STEP 4: STARTING VENV SERVICE ===" -ForegroundColor Cyan
}

Write-Host "Stopping any existing service..." -ForegroundColor Yellow
# Stop both Docker and Python services
ssh "$JetsonUser@$JetsonIP" "pkill -f ml_service.py 2>/dev/null || true"
if ($DeploymentMode -eq "docker" -or $DeploymentMode -eq "docker-local") {
    if ($useSudo) {
        ssh "$JetsonUser@$JetsonIP" "cd ~/hairskin_classifier && echo '$JetsonPassword' | sudo -S docker-compose down 2>/dev/null || true"
    } else {
        ssh "$JetsonUser@$JetsonIP" "cd ~/hairskin_classifier && docker-compose down 2>/dev/null || true"
    }
}

if ($DeploymentMode -eq "docker") {
    Write-Host "Building Docker image remotely..." -ForegroundColor Yellow
    if ($useSudo) {
        $buildResult = ssh "$JetsonUser@$JetsonIP" "cd ~/hairskin_classifier && echo '$JetsonPassword' | sudo -S docker-compose build 2>&1"
    } else {
        $buildResult = ssh "$JetsonUser@$JetsonIP" "cd ~/hairskin_classifier && docker-compose build 2>&1"
    }
} elseif ($DeploymentMode -eq "docker-local") {
    Write-Host "Building Docker image locally..." -ForegroundColor Yellow
    
    # Check if Docker is available locally (WSL2 or Docker Desktop)
    $dockerFound = $false
    
    # Try WSL2 Docker first
    try {
        $wslResult = wsl docker --version 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Docker found in WSL2: $wslResult" -ForegroundColor Green
            $dockerFound = $true
            $useWSL = $true
        }
    } catch {
        # WSL2 Docker not found
    }
    
    # Try local Docker if WSL2 failed
    if (-not $dockerFound) {
        try {
            docker --version | Out-Null
            Write-Host "Docker found locally" -ForegroundColor Green
            $dockerFound = $true
            $useWSL = $false
        } catch {
            # Local Docker not found
        }
    }
    
    if (-not $dockerFound) {
        Write-Host "ERROR: Docker not found!" -ForegroundColor Red
        Write-Host "Please install Docker in WSL2 or Docker Desktop" -ForegroundColor Yellow
        Write-Host "WSL2 Setup:" -ForegroundColor Cyan
        Write-Host "  1. Install WSL2: wsl --install" -ForegroundColor White
        Write-Host "  2. Install Docker in Ubuntu: curl -fsSL https://get.docker.com -o get-docker.sh && sh get-docker.sh" -ForegroundColor White
        Write-Host "  3. Add user to docker group: sudo usermod -aG docker `$USER" -ForegroundColor White
        exit 1
    }
    
    # Build for ARM64 architecture
    Write-Host "Building ARM64 image for Jetson..." -ForegroundColor Yellow
    
    if ($useWSL) {
        # Copy project files to WSL2 temp directory
        $wslTempDir = "/tmp/hairskin_build_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
        Write-Host "Creating WSL2 temp directory: $wslTempDir" -ForegroundColor Yellow
        wsl mkdir -p $wslTempDir
        
        # Copy files to WSL2 (selective - only what Docker needs)
        Write-Host "Copying project files to WSL2..." -ForegroundColor Yellow
        Write-Host "  - Copying source files (excluding large models)..." -ForegroundColor Gray
        
        # Copy only necessary files for Docker build
        wsl cp "ml_service.py" "$wslTempDir/"
        wsl cp "DepiModels.py" "$wslTempDir/"
        wsl cp "requirements.txt" "$wslTempDir/"
        wsl cp "Dockerfile" "$wslTempDir/"
        wsl cp -r "hc_model" "$wslTempDir/"
        wsl cp -r "skin_type_model" "$wslTempDir/"
        
        # Copy DICOM file if exists
        $dicom = Get-ChildItem "*.dcm" | Select-Object -First 1
        if ($dicom) {
            Write-Host "  - Copying test DICOM file..." -ForegroundColor Gray
            wsl cp "$($dicom.Name)" "$wslTempDir/"
        }
        
        Write-Host "  - Essential files copied successfully" -ForegroundColor Green
        
        # Build in WSL2
        Write-Host "Starting Docker build in WSL2 (this may take 10-15 minutes)..." -ForegroundColor Yellow
        Write-Host "  - Building ARM64 image for Jetson Nano..." -ForegroundColor Gray
        Write-Host "  - Progress will be shown below:" -ForegroundColor Gray
        $buildResult = wsl bash -c "cd $wslTempDir && docker buildx build --platform linux/arm64 -t hairskin-classifier:latest . --progress=plain 2>&1"
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "WSL2 build successful, saving image..." -ForegroundColor Green
            
            # Save image to tar file in WSL2
            Write-Host "  - Compressing Docker image (this may take 2-3 minutes)..." -ForegroundColor Gray
            wsl bash -c "cd $wslTempDir && docker save hairskin-classifier:latest | gzip > hairskin-classifier.tar.gz"
            Write-Host "  - Image compressed successfully" -ForegroundColor Green
            
            # Copy tar file back to Windows
            Write-Host "  - Copying compressed image to Windows..." -ForegroundColor Gray
            wsl cp "$wslTempDir/hairskin-classifier.tar.gz" .
            Write-Host "  - Image ready for transfer" -ForegroundColor Green
            
            # Clean up WSL2 temp directory
            Write-Host "  - Cleaning up WSL2 temp files..." -ForegroundColor Gray
            wsl rm -rf $wslTempDir
            Write-Host "  - Cleanup complete" -ForegroundColor Green
        }
    } else {
        # Use local Docker
        $buildResult = docker buildx build --platform linux/arm64 -t hairskin-classifier:latest . 2>&1
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Local build successful, saving image..." -ForegroundColor Green
            # Save image to tar file
            docker save hairskin-classifier:latest | gzip > hairskin-classifier.tar.gz
        }
    }
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Transferring Docker image to Jetson (this may take 3-5 minutes)..." -ForegroundColor Yellow
        Write-Host "  - Uploading compressed image (~500MB)..." -ForegroundColor Gray
        scp hairskin-classifier.tar.gz "$JetsonUser@${JetsonIP}:/tmp/"
        Write-Host "  - Upload complete" -ForegroundColor Green
        
        # Load image on Jetson
        Write-Host "Loading image on Jetson..." -ForegroundColor Yellow
        Write-Host "  - Extracting and loading Docker image..." -ForegroundColor Gray
        ssh "$JetsonUser@$JetsonIP" "docker load < /tmp/hairskin-classifier.tar.gz"
        Write-Host "  - Image loaded successfully" -ForegroundColor Green
        
        Write-Host "  - Cleaning up temp files on Jetson..." -ForegroundColor Gray
        ssh "$JetsonUser@$JetsonIP" "rm /tmp/hairskin-classifier.tar.gz"
        
        # Clean up local tar file
        Write-Host "  - Cleaning up local temp files..." -ForegroundColor Gray
        Remove-Item hairskin-classifier.tar.gz -ErrorAction SilentlyContinue
        
        # Transfer Docker Compose file after successful image transfer
        Write-Host "Transferring Docker Compose file..." -ForegroundColor Yellow
        scp "docker-compose.yml" "$JetsonUser@${JetsonIP}:~/hairskin_classifier/"
        Write-Host "  - Docker Compose file transferred" -ForegroundColor Green
        
        Write-Host "Docker image deployed successfully" -ForegroundColor Green
        $buildResult = "Local build and transfer successful"
    } else {
        Write-Host "Docker build failed!" -ForegroundColor Red
        if ($useWSL) {
            wsl rm -rf $wslTempDir 2>/dev/null
        }
    }
} else {
    Write-Host "Starting Python venv service..." -ForegroundColor Yellow
    ssh "$JetsonUser@$JetsonIP" "cd ~/hairskin_classifier && nohup ./venv/bin/python ml_service.py > service.log 2>&1 &"
    Start-Sleep -Seconds 8
    $serviceMode = "venv"
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker build FAILED!" -ForegroundColor Red
    Write-Host "Error details:" -ForegroundColor Yellow
    Write-Host $buildResult -ForegroundColor Red
    Write-Host "`nTrying direct Python execution instead..." -ForegroundColor Yellow
    
    # Stop any existing Python service first
    ssh "$JetsonUser@$JetsonIP" "pkill -f ml_service.py 2>/dev/null || true"
    Start-Sleep -Seconds 2
    
    # Fallback: Run Python service directly
    Write-Host "Starting ML service with Python 3.7..." -ForegroundColor Yellow
    ssh "$JetsonUser@$JetsonIP" "cd ~/hairskin_classifier && source venv/bin/activate && nohup python ml_service.py > service.log 2>&1 &"
    Start-Sleep -Seconds 8
    
    $healthCheck = ssh "$JetsonUser@$JetsonIP" "curl -s -m 5 http://localhost:8000/health 2>/dev/null || echo 'FAILED'"
    if ($healthCheck -ne "FAILED" -and $healthCheck -ne "") {
        Write-Host "Python service: STARTED" -ForegroundColor Green
        $serviceMode = "python"
    } else {
        Write-Host "Python service: FAILED" -ForegroundColor Red
        $logs = ssh "$JetsonUser@$JetsonIP" "cd ~/hairskin_classifier && tail -10 service.log 2>/dev/null || echo 'No logs available'"
        Write-Host "Service logs:" -ForegroundColor Yellow
        Write-Host $logs -ForegroundColor White
        exit 1
    }
} else {
    Write-Host "Docker build: SUCCESS" -ForegroundColor Green
    
    Write-Host "Starting Docker service..." -ForegroundColor Yellow
    if ($useSudo) {
        $startResult = ssh "$JetsonUser@$JetsonIP" "cd ~/hairskin_classifier && echo '$JetsonPassword' | sudo -S docker-compose up -d 2>&1"
    } else {
        $startResult = ssh "$JetsonUser@$JetsonIP" "cd ~/hairskin_classifier && docker-compose up -d 2>&1"
    }
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Docker service: STARTED" -ForegroundColor Green
        $serviceMode = "docker"
    } else {
        Write-Host "Docker start failed, trying Python fallback..." -ForegroundColor Yellow
        Write-Host "Docker error:" -ForegroundColor Red
        Write-Host $startResult -ForegroundColor Red
        
        # Stop any existing Python service first
        ssh "$JetsonUser@$JetsonIP" "pkill -f ml_service.py 2>/dev/null || true"
        Start-Sleep -Seconds 2
        
        ssh "$JetsonUser@$JetsonIP" "cd ~/hairskin_classifier && source venv/bin/activate && nohup python ml_service.py > service.log 2>&1 &"
        Start-Sleep -Seconds 8
        $serviceMode = "python"
    }
}

# STEP 5: Verify deployment
Write-Host "`n=== STEP 5: VERIFYING DEPLOYMENT ===" -ForegroundColor Cyan

if ($serviceMode -eq "docker") {
    Write-Host "Waiting for Docker service to be ready..." -ForegroundColor Yellow
    Start-Sleep -Seconds 15
} else {
    Write-Host "Waiting for Python service to be ready..." -ForegroundColor Yellow
    Start-Sleep -Seconds 10
}

# Try health check multiple times
$maxRetries = 3
$healthCheck = "FAILED"
for ($i = 1; $i -le $maxRetries; $i++) {
    Write-Host "Health check attempt $i/$maxRetries..." -ForegroundColor Yellow
    $healthCheck = ssh "$JetsonUser@$JetsonIP" "curl -s -m 10 http://localhost:8000/health 2>/dev/null || echo 'FAILED'"
    if ($healthCheck -ne "FAILED" -and $healthCheck -ne "") {
        break
    }
    if ($i -lt $maxRetries) {
        Start-Sleep -Seconds 5
    }
}

if ($healthCheck -ne "FAILED" -and $healthCheck -ne "") {
    Write-Host "Health check: SUCCESS" -ForegroundColor Green
    Write-Host $healthCheck -ForegroundColor White
} else {
    Write-Host "Health check: FAILED" -ForegroundColor Red
    
    # Show process status
    $processes = ssh "$JetsonUser@$JetsonIP" "ps aux | grep -E '(ml_service|docker-compose|python)' | grep -v grep || echo 'No processes found'"
    Write-Host "Running processes:" -ForegroundColor Yellow
    Write-Host $processes -ForegroundColor White
    
    # Show logs
    if ($serviceMode -eq "python") {
        $logs = ssh "$JetsonUser@$JetsonIP" "cd ~/hairskin_classifier && tail -20 service.log 2>/dev/null || echo 'No service logs'"
        Write-Host "Service logs:" -ForegroundColor Yellow
        Write-Host $logs -ForegroundColor White
    } else {
        $logs = ssh "$JetsonUser@$JetsonIP" "cd ~/hairskin_classifier && docker-compose logs --tail=20 2>/dev/null || echo 'No docker logs'"
        Write-Host "Docker logs:" -ForegroundColor Yellow
        Write-Host $logs -ForegroundColor White
    }
}

# Create management scripts
Write-Host "`n=== CREATING MANAGEMENT SCRIPTS ===" -ForegroundColor Cyan

if ($serviceMode -eq "docker") {
    $startScript = @"
#!/bin/bash
echo "Starting HairSkin Classifier (Docker)..."
cd ~/hairskin_classifier
docker-compose up -d
echo "Service started at http://localhost:8000"
"@
    
    $stopScript = @"
#!/bin/bash
echo "Stopping HairSkin Classifier (Docker)..."
cd ~/hairskin_classifier
docker-compose down
echo "Service stopped"
"@
} else {
    $startScript = @"
#!/bin/bash
echo "Starting HairSkin Classifier (Python 3.7)..."
cd ~/hairskin_classifier
source venv/bin/activate
nohup python ml_service.py > service.log 2>&1 &
echo "Service started at http://localhost:8000"
echo "Logs: tail -f ~/hairskin_classifier/service.log"
"@
    
    $stopScript = @"
#!/bin/bash
echo "Stopping HairSkin Classifier (Python 3.7)..."
pkill -f ml_service.py
echo "Service stopped"
"@
}

ssh "$JetsonUser@$JetsonIP" "cat > ~/start_hairskin.sh << 'EOF'
$startScript
EOF"

ssh "$JetsonUser@$JetsonIP" "cat > ~/stop_hairskin.sh << 'EOF'
$stopScript
EOF"

ssh "$JetsonUser@$JetsonIP" "chmod +x ~/start_hairskin.sh ~/stop_hairskin.sh"

Write-Host "`n" + "="*60 -ForegroundColor Green
Write-Host "MASTER DEPLOYMENT COMPLETED!" -ForegroundColor Green
Write-Host "="*60 -ForegroundColor Green
Write-Host ""
Write-Host "Service Mode: $serviceMode" -ForegroundColor Cyan
Write-Host "Service URL: http://$JetsonIP:8000" -ForegroundColor Cyan
Write-Host "API docs: http://$JetsonIP:8000/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "Management Commands:" -ForegroundColor Yellow
Write-Host "  Start: ssh $JetsonUser@$JetsonIP '~/start_hairskin.sh'" -ForegroundColor White
Write-Host "  Stop:  ssh $JetsonUser@$JetsonIP '~/stop_hairskin.sh'" -ForegroundColor White
if ($serviceMode -eq "python") {
    Write-Host "  Logs:  ssh $JetsonUser@$JetsonIP 'tail -f ~/hairskin_classifier/service.log'" -ForegroundColor White
} else {
    Write-Host "  Logs:  ssh $JetsonUser@$JetsonIP 'cd ~/hairskin_classifier && docker-compose logs -f'" -ForegroundColor White
}
Write-Host ""
Write-Host "Project location: ~/hairskin_classifier" -ForegroundColor Yellow
Write-Host "Python venv: ~/hairskin_classifier/venv" -ForegroundColor Yellow
Write-Host ""
