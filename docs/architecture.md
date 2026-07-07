# Architecture

## System Overview

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Client     │────▶│  SGLang API  │────▶│  JAX Engine │
│  (Python)    │     │   Server     │     │   (XLA)     │
└─────────────┘     └──────────────┘     └─────────────┘
                                               │
                                          ┌────┴────┐
                                          │         │
                                        TPU       GPU
```

## Components

### Engine
The core inference engine that manages model loading, tokenization, and generation.

### Scheduler
Handles request batching and scheduling for optimal throughput.

### Cache
KV-cache management with RadixCache for prefix sharing across requests.

### Server
HTTP/gRPC server exposing the SGLang API for client applications.

## Data Flow

1. Client sends a generation request via HTTP/gRPC
2. Server validates and enqueues the request
3. Scheduler batches compatible requests
4. Engine runs JAX-compiled inference on TPU/GPU
5. Results stream back to the client

