## Running vLLM with usage billing

This proof of concept modifies our vLLM helm chart to enable exporting of metrics and traces to be used for metered billing.

It includes:

- Prometheus as a dependency for metrics - adding needed annotations for vLLM deployment
- OpenTelemetry Collector to receive and export traces
- Clickhouse database to store traces


This POC was executed in a minikube cluster. You can run it by following the instructions below:

Start Minikube. Nvidia GPU recommended as this example uses a CUDA optimized image

```bash
minikube start --driver docker --container-runtime docker --gpus all --memory 10g --cpus 6
```

Compile a container image based on vLLM image. This is needed because their image does not come with the necessary OpenTelemetry libraries. Note that we tried to install the exact versions in [their documentation](https://github.com/vllm-project/vllm/blob/main/examples/production_monitoring/Otel.md), but those did not export traces when using OpenTelemetry FastAPI Auto Instrumentation, so we removed the version restrictions.

```bash
make build-minikube
```

Install the helm chart

```bash
export HUGGING_FACE_HUB_TOKEN=<YOUR-TOKEN-HERE>
cd charts
helm install vllm llm-inference-app --set env.HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}
```

## Examples

After getting the application running, we can run some examples.

Expose the application with port forwarding

```bash
kubectl port-forward services/vllm-llm-inference-app 8000:8000
```

Send a request to the application

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -H "traceparent: 00-d5fe1dc9035165ce36952daf29686b6c-14330be33197dd1a-01" \
    -H "X-Request-Id: some-user-request-id" \
    -d '{
        "model": "facebook/opt-125m",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }'
```

You should see a result similar to this:

```json
{
  "id": "cmpl-f36cd2007ac1456490116423fa3a5f4a",
  "object": "text_completion",
  "created": 1733248945,
  "model": "facebook/opt-125m",
  "choices": [
    {
      "index": 0,
      "text": " great place to live.  I",
      "logprobs": null,
      "finish_reason": "length",
      "stop_reason": null,
      "prompt_logprobs": null
    }
  ],
  "usage": {
    "prompt_tokens": 5,
    "total_tokens": 12,
    "completion_tokens": 7,
    "prompt_tokens_details": null
  }
}
```

To check if data is being sent to the database, we can port-forward Clickhouse with:

```bash
kubectl port-forward services/vllm-clickhouse 9000:9000
```

Access the database using Clickhouse client:

```bash
clickhouse client --password default-password
```

Run query in console:

```sql
select * from otel_traces;
```

You should see the traces produced by the application with the same TraceId that we passed to the request as the `traceparent` header.