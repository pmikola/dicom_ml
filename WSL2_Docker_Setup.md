# WSL2 + Docker Setup for Jetson Deployment

## Quick Setup Commands

### 1. Install WSL2 (if not already installed)
```powershell
# Run as Administrator
wsl --install
# Restart computer when prompted
```

### 2. Install Docker in WSL2 Ubuntu
```bash
# Step 1: Enter WSL2
wsl

# Step 2: Set WSL2 user password (if not already set)
sudo passwd $USER
# Enter new password twice when prompted

# Step 3: Download Docker install script
curl -fsSL https://get.docker.com -o get-docker.sh

# Step 4: Install Docker (will ask for password)
sudo sh get-docker.sh

# Step 5: Add user to docker group
sudo usermod -aG docker $USER

# Step 6: Start Docker service
sudo service docker start

# Step 7: Test Docker installation
docker --version

# Step 8: Exit WSL2
exit
```

### 3. Enable Docker Buildx (for ARM64 builds)
```bash
# In WSL2 Ubuntu
# Install QEMU emulators for cross-platform building
docker run --privileged --rm tonistiigi/binfmt --install all

# Create multi-platform builder
docker buildx create --use --name multiarch --driver docker-container

# Bootstrap the builder
docker buildx inspect --bootstrap

# Verify ARM64 support
docker buildx ls
# Should show: linux/amd64, linux/arm64, etc.
```

### 4. Test the Setup
```powershell
# From Windows PowerShell
wsl docker --version
# Should show Docker version

# Test ARM64 build capability
wsl docker buildx ls
# Should show multiarch builder with linux/arm64 support
```

### 5. Prepare for Jetson Nano Builds
```bash
# In WSL2 Ubuntu - create temp directory for builds
mkdir -p /tmp/jetson_builds

# Test a simple ARM64 build
echo "FROM alpine:latest" > /tmp/test_dockerfile
echo "RUN echo 'ARM64 test successful'" >> /tmp/test_dockerfile

# Test ARM64 build (should work without errors)
docker buildx build --platform linux/arm64 -f /tmp/test_dockerfile -t test-arm64 .

# Clean up test
rm /tmp/test_dockerfile
docker rmi test-arm64 2>/dev/null || true
```

## Usage with Deployment Script

Once setup is complete, use:
```powershell
.\deploy_jetson_master.ps1 -DeploymentMode "docker-local"
```

### What the Script Does Automatically:

1. **Detects WSL2 Docker** with ARM64 support
2. **Creates temp directory** in WSL2 (`/tmp/hairskin_build_TIMESTAMP`)
3. **Copies only essential files** to WSL2 (no large model files)
4. **Builds ARM64 Docker image** for Jetson Nano architecture
5. **Compresses image** to tar.gz (~500MB)
6. **Transfers to Jetson** via SCP
7. **Loads and starts** the containerized service

### Expected Timeline:
- **File copy to WSL2**: 10-30 seconds
- **Docker build**: 5-10 minutes (on i9 CPU)
- **Image compression**: 1-2 minutes
- **Transfer to Jetson**: 2-5 minutes (depending on network)
- **Load on Jetson**: 30-60 seconds
- **Total**: ~10-15 minutes

## Troubleshooting

### Common Issues:

**"sudo: command not found" or password prompts**
- First set your WSL2 password: `sudo passwd $USER`
- Enter the same password when prompted for sudo commands

**"docker: command not found"**
- Make sure you ran `sudo sh get-docker.sh` (not just the curl command)
- Check if Docker service is running: `sudo service docker start`

**"Unit docker.service not found"**
- Docker installation failed or incomplete
- Re-run: `sudo sh get-docker.sh`

**"permission denied" when running docker**
- Add user to docker group: `sudo usermod -aG docker $USER`
- Restart WSL2: `exit` then `wsl` again

## Benefits

✅ **No Docker Desktop needed**  
✅ **Native Linux Docker environment**  
✅ **Better ARM64 cross-compilation**  
✅ **Faster builds than remote building**  
✅ **Full control over Docker configuration**
