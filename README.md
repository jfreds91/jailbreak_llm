# AMD GPU + PyTorch Setup Guide for Ubuntu 24.04

## Your System Configuration
- **OS**: Ubuntu 24.04 (Noble)
- **GPU**: AMD Radeon RX 9700
- **CPU**: Intel Core i7-14700K
- **Required ROCm Version**: 6.2+ (your GPU is too new for older versions)
- **Required PyTorch**: ROCm 6.4 build with Python 3.11

---

## Step 1: Install ROCm 6.2 Runtime

### Check if ROCm is installed:
```bash
# Check ROCm packages
apt list --installed | grep -E 'rocm|hip|hsa'

# Try to run ROCm tools
/opt/rocm/bin/rocm-smi --version
/opt/rocm/bin/rocminfo | grep "Name:" -A 5
```

### If NOT installed or wrong version:

```bash
# 1. Remove any existing ROCm packages
sudo apt remove --purge 'rocm*' 'hip*' 'hsa*' rocminfo

# 2. Add AMD's ROCm 6.2 repository
sudo mkdir -p /etc/apt/keyrings
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null

echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.2 noble main" | sudo tee /etc/apt/sources.list.d/rocm.list

# 3. Update and install ROCm runtime
sudo apt update
sudo apt install rocm-core hip-runtime-amd rocm-device-libs hsa-rocr hsakmt-roct-dev

# 4. Add yourself to required groups
sudo usermod -a -G render,video $LOGNAME

# 5. Add ROCm to PATH
echo 'export PATH=/opt/rocm/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# 6. REBOOT (required for group changes and driver loading)
sudo reboot
```

### After reboot, verify:
```bash
rocm-smi
rocminfo | grep "Name:" -A 5
```

**Expected**: `rocminfo` should show BOTH your CPU and GPU as HSA agents. If it only shows CPU, ROCm doesn't recognize your GPU.

---

## Step 2: Install PyTorch with ROCm 6.4

### Why Use pip Instead of uv?

**The Problem**: Ubuntu 24.04 reports itself as `manylinux_2_39` (glibc 2.39), but PyTorch ROCm wheels are built for `manylinux_2_28`. This is actually **fine** - newer glibc versions are backward compatible.

**uv's Issue**: `uv` is overly strict about platform tags and refuses to install `manylinux_2_28` wheels on `manylinux_2_39` systems.

**pip's Advantage**: Regular `pip` is more lenient and will install the wheels anyway, recognizing the backward compatibility.

**Alternative**: You could also configure uv to be less strict, but using pip directly is simpler for this edge case.

### Installation Steps:

```bash
# Navigate to your project directory
cd ~/your-project

# Create Python 3.11 virtual environment with uv
uv venv --python 3.11
python3 -m ensurepip

# Activate the venv
source .venv/bin/activate

# Use pip (from within the venv) to install PyTorch
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4

# Install other dependencies
python3 -m pip install ipykernel matplotlib numpy pandas heretic-llm

# OR install other dependencies with uv
uv pip install ipykernel matplotlib numpy pandas heretic-llm
```

### Verify Installation:

```bash
.venv/bin/python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'ROCm version: {torch.version.hip}')
print(f'GPU available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
print(f'GPU name: {torch.cuda.get_device_name(0)}')

# Test actual computation
x = torch.randn(1000, 1000).cuda()
y = torch.randn(1000, 1000).cuda()
z = torch.matmul(x, y)
print(f'Computation successful on: {z.device}')
"
```

**Expected output**:
```
PyTorch version: 2.9.1+rocm6.4
ROCm version: 6.4
GPU available: True
GPU count: 1
GPU name: AMD Radeon RX 9700 XT
Computation successful on: cuda:0
```

### Set up HuggingFace to use the right cache location. It's going to store a lot of data here, so make sure it's a big drive
```export HF_HOME=/home/bigdrive/.cache```

### Run heretic
```heretic Qwen/Qwen3-14B-FP8```

### register into pykernel
```python -m ipykernel install --user --name=jailbreak_llm --display-name "Python (jailbreak_llm)"```

---

## Step 3: Project Configuration (Optional)

If you want to use `pyproject.toml` with uv, you'll need to install torch manually as shown above, then let uv manage other dependencies:

```toml
[project]
name = "your-project"
version = "0.1.0"
requires-python = ">=3.11,<3.12"
dependencies = [
    "ipykernel>=7.1.0",
    "matplotlib",
    "numpy",
    "pandas",
    # Don't include torch/torchvision here - install manually with pip
]
```

Then:
```bash
# Create venv and install other dependencies
uv sync

# Separately install torch with pip
.venv/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4
```

---

## Common Issues & Fixes

### Issue: "The GPU node has an unrecognized id"
**Cause**: ROCm version too old for your GPU  
**Fix**: Upgrade to ROCm 6.2+ (your RX 9700 is too new for ROCm 5.7)

### Issue: Mix of ROCm 5.7 and 6.2 packages
**Cause**: Ubuntu's default repos have ROCm 5.7; AMD's repo has 6.2  
**Fix**: Always specify exact versions or use AMD's repo exclusively  
```bash
sudo apt install rocminfo=1.0.0.60200-66~24.04
```

### Issue: `torch.cuda.is_available()` returns False
**Checks**:
1. Are you in the `render` and `video` groups? (`groups` command)
2. Did you reboot after adding groups?
3. Does `rocminfo` show your GPU?
4. Is PyTorch the ROCm build? (check `torch.version.hip`)

### Issue: Warning about `/opt/amdgpu/share/libdrm/amdgpu.ids`
**Fix** (optional - this is cosmetic):
```bash
sudo mkdir -p /opt/amdgpu/share/libdrm
sudo ln -s /usr/share/libdrm/amdgpu.ids /opt/amdgpu/share/libdrm/amdgpu.ids
```

---

## Version Compatibility Summary

| Component | Version | Why |
|-----------|---------|-----|
| ROCm | 6.2+ | RX 9700 too new for 5.7 |
| PyTorch | 2.9.1+rocm6.4 | Latest with ROCm support |
| Python | 3.11 | PyTorch ROCm 6.4 doesn't support 3.13 yet |
| glibc | 2.39 (Ubuntu 24.04) | Compatible with manylinux_2_28 wheels |

---

## Quick Verification Checklist

- [ ] `rocm-smi` shows your GPU
- [ ] `rocminfo` shows GPU as HSA agent (not just CPU)
- [ ] `groups` shows you're in `render` and `video`
- [ ] `torch.__version__` contains `rocm6.4`
- [ ] `torch.cuda.is_available()` returns `True`
- [ ] Can create tensors on GPU: `torch.tensor([1.0]).cuda()`

# Notes

### FP8 weights
- 1B FP8 weights is 1gb plus some overhead. A 16gb GPU can load a 14B FP8 model, but it will not have much room to allocate for the KV cache, so we have to be careful 
- FP8 weights are getting upcast dynamically to bf16 in order to do matrix math. This is happening in a custom FP8LinearLayer.
    - this is necessary because until ROCm 6.5, FP8 operations are not supported. The latest Pytorch version is compiled against ROCm 6.4. The latest ROCm library version is 7.1 and is backward compatible with pytorch
- I modified heretic (in site-packages) to determine which layers are abliterable, and to cast them to bf16 automatically, just one time. This increases RAM requirements by 30%. We save the mixed-precision model

### GGUF
- note I had to set the LLAMA_CACHE env var to make it save to the right place.
- I also tried downloading a 14B, and then 32B, parameter model using GGUF. This is a single-file format which is used with `llama.cpp`. GGUF supports multiple quantizations, and I opted for a FP8 one (Q8_0). I was able to run both locally with mixed results:
- 14B: I found that I could fit 30 layers in vRAM, if I quantized the KV-cache. I was able to get about 20 tokens/s output this way. The suggested context window of 40k tokens makes the KV cache way too big (it just arena allocates) so I did a very small one of 8k. The model did not refused prompts readily, and the answers were ok, but it would get the system prompt confused with the user prompt (I told it to pretend to be a 9 year old girl who didn't know how to code, and it gave me code answers explained to me as if I were 9 - CoT explicitly stated the USER was 9 and didn't know code). GPU was 99% vRAM saturated and 100% utilized.

```bash
./build/bin/llama-cli -hf Qwen/Qwen3-14B-GGUF:Q8_0 --jinja --color on  -ngl 30 -fa on -sm row --temp 0.6 --top-k 20 --top-p 0.95 --min-p 0 --presence-penalty 1.5 -c 16384 -n 8192 --no-context-shift -ctk q8_0 -ctv q8_0 -sys "You are a helpful assistant"

```

- 32B: At this size I could fit 20 layers in vRAM (75% vRAM saturated) but there is a massive bottleneck in populating the KV cache. I had to do a few things to relieve it: (1) explicitly tell llama.cpp to utilize many cores (I have 28 cores). (2) Lower the batch size from 2048 to 512 (not sure if this was truly needed). (3) most critical - remove KV quantization. The quant/dequant op appears to be single-threaded, so when this was required, most cores sat idle. Removing this immediately allowed me to leverage all cores. I was able to get 2 tokens/s output this way. Probably could move another layer or two into vRAM and get more like 4, but it's still unusable. The model was more willing to refuse prompts.

```bash
./build/bin/llama-cli -hf Qwen/Qwen3-32B-GGUF:Q8_0 --jinja --color on  -ngl 20 -fa on -sm row --temp 0.6 --top-k 20 --top-p 0.95 --min-p 0 --presence-penalty 1.5 -c 4096 -n 8192 --no-context-shift -sys "You are a helpful assistant" -t 25 -tb 25 -b 512 -ub 512
```