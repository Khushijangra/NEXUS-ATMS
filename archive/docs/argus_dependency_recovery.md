# ARGUS Dependency Recovery

## 1. Dependency Declarations
The missing dependencies for ARGUS are explicitly declared in:
`argus_stream_extracted\argus stream A\requirements.txt`

The file lists:
- `timm`
- `transformers`
- `gradio` (for demo)

## 2. Installation Command
To install the dependencies into the root NEXUS environment without modifying the codebase:
```powershell
pip install -r "argus_stream_extracted\argus stream A\requirements.txt"
```
Or manually:
```powershell
pip install timm transformers gradio
```

## 3. Package Compatibility Verification
- **Python Version:** 3.13.7
- **PyTorch Version:** 2.6.0+cu124
- **Compatibility:** Both `timm` and Hugging Face `transformers` are actively maintained and fully support Python 3.13 and PyTorch 2.6. There are no ABI compilation requirements since they are pure Python wrappers around PyTorch primitives, ensuring safe recovery in this local environment.
