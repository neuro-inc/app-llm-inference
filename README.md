# vLLM inference app

Deploy vLLM [vLLM](https://github.com/vllm-project/vllm) Apolo app.

## Platform deploymet example:
This example deploys `lmsys/vicuna-7b-v1.3` LLM model.

```yaml
apolo run --pass-config ghcr.io/neuro-inc/app-deployment -- install https://github.com/neuro-inc/app-llm-inference \
  llm-inference vicuna7b charts/llm-inference-app \
  --timeout=10m \
  --set "model.modelHFName=lmsys/vicuna-7b-v1.3" \
  --set "model.tokenizerHFName=lmsys/vicuna-7b-v1.3" \
  --set "model.modelRevision=720244025c1a7e15661a174c63cce63c8218e52b" \ # optional
  --set "model.tokenizerRevision=720244025c1a7e15661a174c63cce63c8218e52b" \ # optional
  --set "preset_name=H100x1" \  # set needed preset
  --set 'serverExtraArgs[0]=--dtype=half' \ # optional
  --set "env.HUGGING_FACE_HUB_TOKEN=YOUR_TOKEN" \ # optional
  --set "ingress.enabled=True" \ # optional
  --set "ingress.clusterName=scottdc" # optional
```

## Helm install
```
helm install llm-inference-app . \
  --timeout=10m \
  --set "model.modelHFName=lmsys/vicuna-7b-v1.3" \
  --set "model.tokenizerHFName=lmsys/vicuna-7b-v1.3" \
  --set "env.HUGGING_FACE_HUB_TOKEN=YOUR_TOKEN \
  --set "ingress.enabled=true" \
  --set "ingress.clusterName=novoserve"
```
