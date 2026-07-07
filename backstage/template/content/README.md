# ${{ values.name }}

${{ values.description }}

## Model

- **Model**: `${{ values.model_name }}`
- **TPU**: `${{ values.tpu_type }}`

## Quick Start

```bash
pip install -r requirements.txt
python serve.py --model ${{ values.model_name }} --tpu-type ${{ values.tpu_type }}
```

## API

The service exposes an OpenAI-compatible API:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "${{ values.model_name }}",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```
