FROM vllm/vllm-openai:v0.6.4
# FROM vllm/vllm-openai:v0.6.4.post1

RUN python3 -m pip install \
    'opentelemetry-sdk' \
    'opentelemetry-api' \
    'opentelemetry-exporter-otlp' \
    'opentelemetry-semantic-conventions-ai' \
    'opentelemetry-instrumentation-fastapi' \
    "opentelemetry-distro"
    # 'opentelemetry-sdk>=1.26.0,<1.27.0' \
    # 'opentelemetry-api>=1.26.0,<1.27.0' \
    # 'opentelemetry-exporter-otlp>=1.26.0,<1.27.0' \
    # 'opentelemetry-semantic-conventions-ai>=0.4.1,<0.5.0' \
    # 'opentelemetry-instrumentation-fastapi' \
    # "opentelemetry-distro"

# RUN opentelemetry-bootstrap --action=install

ENTRYPOINT [ "opentelemetry-instrument", "--traces_exporter", "console,otlp", "--exporter_otlp_protocol", "http/protobuf", "--metrics_exporter", "none", "--log_level", "debug","python3", "-m", "vllm.entrypoints.openai.api_server" ]
