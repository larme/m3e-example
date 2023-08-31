m3e model embedding and ranking endpoints. Ranking is done in API worker so it's better to spin as many api workers as possible.

```
python3 -m venv venv && . venv/bin/activate
pip install -r requirements.txt
bentoml serve service:svc --api-workers=8
```

To run 4 copies of m3e model on the first GPU in parallel with 8 api workers:

```
BENTOML_CONFIG_OPTIONS='runners.m3e_runner.workers_per_resource=4 runners.m3e_runner.resources."nvidia.com/gpu"[0]=0' bentoml serve service:svc --api-workers=8
```

The following command utilize the second and the forth GPU, each GPU has 10 copies of m3e model, and also run 32 api workers:

```
BENTOML_CONFIG_OPTIONS='runners.m3e_runner.workers_per_resource=10 runners.m3e_runner.resources."nvidia.com/gpu"[0]=1 runners.m3e_runner.resources."nvidia.com/gpu"[1]=3' bentoml serve service:svc --api-workers=32
```
