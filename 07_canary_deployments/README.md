# Topic 7: Canary Deployments

## What You'll Learn

This topic teaches you how to:
- Deploy new model versions safely
- Split traffic between versions
- Gradually roll out updates
- Rollback quickly if issues occur
- A/B test different models
- Monitor canary performance

## Why Canary Deployments?

### Benefits
- **Risk reduction**: Test new version on small traffic
- **Gradual rollout**: Increase confidence before full deployment
- **Quick rollback**: Revert if issues detected
- **A/B testing**: Compare model versions
- **Zero downtime**: No service interruption

### When to Use
- **Model updates**: New model version
- **Configuration changes**: Different parameters
- **A/B testing**: Compare model performance
- **Risk mitigation**: Critical production systems

## Canary Deployment Strategies

### 1. Traffic Splitting
Route X% of traffic to new version.

### 2. User-based
Route specific users to new version.

### 3. Header-based
Route based on HTTP headers.

### 4. Gradual Rollout
Start with 5%, increase to 100% over time.

## Implementation Options

### Option 1: Kubernetes with Service Mesh (Istio)
Most flexible, supports advanced routing.

### Option 2: Kubernetes with Multiple Deployments
Simple, uses native K8s features.

### Option 3: Application-level Routing
Handle in application code.

## Simple Canary with K8s

### Step 1: Deploy Stable Version
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-serving-stable
spec:
  replicas: 9
  selector:
    matchLabels:
      app: llm-serving
      version: stable
```

### Step 2: Deploy Canary Version
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-serving-canary
spec:
  replicas: 1
  selector:
    matchLabels:
      app: llm-serving
      version: canary
```

### Step 3: Service Selector
Service routes to both versions:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: llm-serving
spec:
  selector:
    app: llm-serving  # Matches both stable and canary
```

Traffic splits 90/10 based on replica count.

## Istio Canary (Advanced)

### VirtualService
Control traffic routing:
```yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: llm-serving
spec:
  hosts:
  - llm-serving
  http:
  - match:
    - headers:
        canary:
          exact: "true"
    route:
    - destination:
        host: llm-serving
        subset: canary
      weight: 100
  - route:
    - destination:
        host: llm-serving
        subset: stable
      weight: 90
    - destination:
        host: llm-serving
        subset: canary
      weight: 10
```

### DestinationRule
Define subsets:
```yaml
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: llm-serving
spec:
  host: llm-serving
  subsets:
  - name: stable
    labels:
      version: stable
  - name: canary
    labels:
      version: canary
```

## Gradual Rollout

### Phase 1: 5% Traffic (Day 1)
- Deploy canary with 1 replica
- Monitor for 24 hours
- Check metrics: latency, errors, throughput

### Phase 2: 25% Traffic (Day 2)
- Scale canary to 3 replicas
- Continue monitoring

### Phase 3: 50% Traffic (Day 3)
- Scale canary to 5 replicas
- Monitor closely

### Phase 4: 100% Traffic (Day 4)
- Scale canary to 10 replicas
- Remove stable deployment
- Canary becomes new stable

## Monitoring Canary

### Key Metrics to Compare
- **Latency**: P50, P95, P99
- **Error rate**: Failed requests
- **Throughput**: Requests per second
- **GPU utilization**: Resource usage
- **Model-specific**: Tokens/second, quality metrics

### Prometheus Queries
```promql
# Compare latency
histogram_quantile(0.95, 
  rate(llm_request_duration_seconds_bucket{version="canary"}[5m])
) / 
histogram_quantile(0.95, 
  rate(llm_request_duration_seconds_bucket{version="stable"}[5m])
)

# Compare error rates
rate(llm_requests_total{version="canary",status="500"}[5m]) /
rate(llm_requests_total{version="canary"}[5m])
```

## Rollback Procedure

### Automatic Rollback
Set up alerts that trigger rollback:
- Error rate > threshold
- Latency > threshold
- Custom metrics fail

### Manual Rollback
```bash
# Scale canary to 0
kubectl scale deployment llm-serving-canary --replicas=0

# Or route all traffic to stable
# Update VirtualService weights
```

## A/B Testing

### Compare Models
Deploy two different models:
- Model A: Current production
- Model B: New candidate

### Metrics to Track
- **Performance**: Latency, throughput
- **Quality**: User feedback, accuracy
- **Cost**: GPU hours, inference cost

### Decision Criteria
- Better performance AND quality → Promote
- Worse performance OR quality → Reject
- Mixed results → Continue testing

## Best Practices

1. **Start small**: 5-10% traffic initially
2. **Monitor closely**: Watch metrics during rollout
3. **Have rollback plan**: Know how to revert quickly
4. **Test thoroughly**: Test canary before production
5. **Document changes**: Track what changed
6. **Communicate**: Inform team of canary deployment

## Common Issues

### Canary Performing Worse
- Check if it's expected (new model might be slower)
- Compare metrics carefully
- Consider rollback if critical

### Traffic Not Splitting Correctly
- Verify service selectors
- Check replica counts
- Verify routing rules (if using Istio)

### Canary Not Getting Traffic
- Check service endpoints
- Verify labels match
- Check routing configuration

## Exercises

1. **Basic canary**: Deploy canary with 10% traffic
2. **Gradual rollout**: Increase from 10% to 100% over phases
3. **Monitoring**: Set up dashboards to compare versions
4. **Rollback**: Practice rolling back canary
5. **A/B test**: Compare two different models

## Next Steps

- **Topic 8**: Monitor canary with Grafana
- **Topic 9**: Model versioning system
- **Topic 10**: Detect drift in canary

## Further Reading

- [Kubernetes Canary Deployments](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/#rolling-back-a-deployment)
- [Istio Traffic Management](https://istio.io/latest/docs/concepts/traffic-management/)
- [Canary Deployment Best Practices](https://www.spinnaker.io/concepts/deployment-strategies/)

