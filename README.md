# vLLM inference app

Deploy vLLM [vLLM](https://github.com/vllm-project/vllm) Apolo app.

## Platform deploymet example:
This example deploys `lmsys/vicuna-7b-v1.3` LLM model.

```yaml
apolo run --pass-config ghcr.io/neuro-inc/app-deployment -- install https://github.com/neuro-inc/app-llm-inference \
  llm-inference vicuna7b charts/llm-inference-app \
  --set timeout=600 \
  --set "llm.modelHFName=lmsys/vicuna-7b-v1.3" \
  --set "llm.tokenizerHFName=lmsys/vicuna-7b-v1.3" \
  --set 'serverExtraArgs[0]=--dtype=half' \
  --set "preset_name=H100x1" \  # set needed preset
  --set "ingress.enabled=True" \ # optional
  --set "ingress.clusterName=scottdc" # optional
```
