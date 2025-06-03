# Installation 
If `uv sync` fails, you probably need to install golang first in order to compile it.

First install golang:
```bash
sudo apt-get install golang-go
```
Then you can run : 
```bash
go install golang.org/dl/go1.24.3@latest
```

Then you should be able to run `uv sync` without any issues.


# Download models
```bash
bash download_iceberg_ckpt.sh
```

# Run the evaluation
```bash
uv run run_evaluation.py
```