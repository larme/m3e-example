Quickly spin up an api endpoint for transformers model. Not the best practice now because we run the model in api server. After turning the model into runner we can have as many api workers as we want.

```
python3 -m venv venv && . venv/bin/activate
pip install -r requirements.txt
bentoml serve service:svc --api-workers=1
```
