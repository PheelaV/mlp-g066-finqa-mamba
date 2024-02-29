Triton is problematic when trying to train the quantized 2.8B model so it fits on our RTX 3090s


version 2.1

```bash
conda create --name triton python=3.11
conda activate triton

git clone https://github.com/openai/triton.git;
```


```
nvim python/triton/runtime/autotuner.py
```
replace 

```
full_nargs = {**self.nargs, **current}
```

with

```
full_nargs = {}
if self.nargs:
    full_nargs.update(self.nargs)
if current:
    full_nargs.update(current)
```
install triton
```bash
git checkout release/v2.1.x
pip install cmake; # build-time dependency
cd triton/python;
pip install -e .
```


previously I have tried this but that installs the newest version of the Triton package:

```
conda create --name triton python=3.11
conda activate triton

git clone https://github.com/openai/triton.git;
cd triton;

pip install ninja cmake wheel; # build-time dependencies
pip install -e python
```

