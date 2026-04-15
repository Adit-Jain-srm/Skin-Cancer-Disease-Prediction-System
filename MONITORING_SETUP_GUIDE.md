"""
Monitoring & Observability Setup Guide
Prometheus + Grafana for the Skin Cancer Detection API

This file contains Docker Compose setup for monitoring infrastructure.
Save as 'monitoring-compose.yml' and run with:
    docker-compose -f monitoring-compose.yml up -d
"""

# ============================================================================
# DOCKER COMPOSE FOR MONITORING STACK
# ============================================================================

# Save this content as 'monitoring-compose.yml' in your project root:

# version: '3.9'
#
# services:
#   prometheus:
#     image: prom/prometheus:latest
#     container_name: prometheus
#     ports:
#       - "9090:9090"
#     volumes:
#       - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
#       - prometheus-data:/prometheus
#     command:
#       - '--config.file=/etc/prometheus/prometheus.yml'
#       - '--storage.tsdb.path=/prometheus'
#       - '--storage.tsdb.retention.time=30d'
#     networks:
#       - monitoring
#     restart: unless-stopped
#
#   grafana:
#     image: grafana/grafana:latest
#     container_name: grafana
#     ports:
#       - "3000:3000"
#     environment:
#       - GF_SECURITY_ADMIN_PASSWORD=admin
#       - GF_INSTALL_PLUGINS=redis-datasource
#     volumes:
#       - grafana-data:/var/lib/grafana
#       - ./monitoring/grafana/provisioning:/etc/grafana/provisioning:ro
#     depends_on:
#       - prometheus
#     networks:
#       - monitoring
#     restart: unless-stopped
#
#   alertmanager:
#     image: prom/alertmanager:latest
#     container_name: alertmanager
#     ports:
#       - "9093:9093"
#     volumes:
#       - ./monitoring/alertmanager.yml:/etc/alertmanager/alertmanager.yml:ro
#       - alertmanager-data:/alertmanager
#     command:
#       - '--config.file=/etc/alertmanager/alertmanager.yml'
#       - '--storage.path=/alertmanager'
#     networks:
#       - monitoring
#     restart: unless-stopped
#
# networks:
#   monitoring:
#     driver: bridge
#
# volumes:
#   prometheus-data:
#     driver: local
#   grafana-data:
#     driver: local
#   alertmanager-data:
#     driver: local

# ============================================================================
# PROMETHEUS CONFIGURATION
# ============================================================================

# Save as 'monitoring/prometheus.yml':

# global:
#   scrape_interval: 15s
#   evaluation_interval: 15s
#   external_labels:
#     monitor: 'skin-cancer-api'
#
# scrape_configs:
#   # Prometheus itself
#   - job_name: 'prometheus'
#     static_configs:
#       - targets: ['localhost:9090']
#
#   # Skin Cancer API
#   - job_name: 'skin-cancer-api'
#     metrics_path: '/metrics'
#     static_configs:
#       - targets: ['localhost:5000']
#     scrape_interval: 10s
#
#   # Docker daemon (if monitoring host)
#   - job_name: 'docker'
#     static_configs:
#       - targets: ['localhost:9323']

# ============================================================================
# ALERTMANAGER CONFIGURATION
# ============================================================================

# Save as 'monitoring/alertmanager.yml':

# global:
#   resolve_timeout: 5m
#   slack_api_url: 'YOUR_SLACK_WEBHOOK_URL'
#
# route:
#   receiver: 'default'
#   group_by: ['alertname', 'cluster', 'service']
#   group_wait: 10s
#   group_interval: 10s
#   repeat_interval: 12h
#   routes:
#     - receiver: 'critical'
#       match:
#         severity: 'critical'
#       continue: true
#
# receivers:
#   - name: 'default'
#     slack_configs:
#       - channel: '#alerts'
#         title: '{{ .GroupLabels.alertname }}'
#         text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'
#
#   - name: 'critical'
#     slack_configs:
#       - channel: '#critical-alerts'
#         title: 'CRITICAL: {{ .GroupLabels.alertname }}'
#         text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'

# ============================================================================
# PROMETHEUS ALERT RULES
# ============================================================================

# Save as 'monitoring/alerts.yml':

# groups:
#   - name: skin_cancer_api
#     interval: 30s
#     rules:
#       # API is down
#       - alert: APIDown
#         expr: up{job="skin-cancer-api"} == 0
#         for: 1m
#         labels:
#           severity: critical
#         annotations:
#           summary: "Skin Cancer API is down"
#           description: "API at {{ $labels.instance }} has been unreachable for >1m"
#
#       # High error rate
#       - alert: HighErrorRate
#         expr: |
#           (sum(rate(http_requests_total{status=~"5.."}[5m])) by (job) /
#            sum(rate(http_requests_total[5m])) by (job)) > 0.05
#         for: 5m
#         labels:
#           severity: warning
#         annotations:
#           summary: "High error rate detected"
#           description: "Error rate is {{ $value | humanizePercentage }}"
#
#       # High latency
#       - alert: HighLatency
#         expr: |
#           histogram_quantile(0.95,
#             sum(rate(http_request_duration_seconds_bucket[5m])) by (le)
#           ) > 0.5
#         for: 5m
#         labels:
#           severity: warning
#         annotations:
#           summary: "High request latency"
#           description: "95th percentile latency is {{ $value | humanizeDuration }}"
#
#       # High memory usage
#       - alert: HighMemoryUsage
#         expr: |
#           (container_memory_usage_bytes{name="skin-cancer-api"} /
#            container_spec_memory_limit_bytes{name="skin-cancer-api"}) > 0.9
#         for: 5m
#         labels:
#           severity: warning
#         annotations:
#           summary: "High memory usage"
#           description: "Memory usage is {{ $value | humanizePercentage }}"
#
#       # High CPU usage
#       - alert: HighCPUUsage
#         expr: |
#           (rate(container_cpu_usage_seconds_total{name="skin-cancer-api"}[5m]) *
#            count(count(container_cpu_usage_seconds_total{name="skin-cancer-api"}) by (cpu))) > 0.8
#         for: 5m
#         labels:
#           severity: warning
#         annotations:
#           summary: "High CPU usage"
#           description: "CPU usage is {{ $value | humanizePercentage }}"

# ============================================================================
# GRAFANA DASHBOARD JSON
# ============================================================================

# Save as 'monitoring/grafana/provisioning/dashboards/skin-cancer-api.json':
# (Use Grafana dashboard import or community dashboards as template)

# Key metrics to include:
# 1. Request Rate (requests/second)
# 2. Error Rate (%)
# 3. Latency (p50, p95, p99)
# 4. Throughput (predictions/second)
# 5. Memory Usage (%)
# 6. CPU Usage (%)
# 7. Model Inference Time
# 8. Queue Depth (if async)

# ============================================================================
# SETUP INSTRUCTIONS
# ============================================================================

"""
1. Create monitoring directory structure:
   mkdir -p monitoring/grafana/provisioning/{dashboards,datasources}

2. Create configuration files (see above sections)

3. Start monitoring stack:
   docker-compose -f monitoring-compose.yml up -d

4. Access services:
   - Prometheus: http://localhost:9090
   - Grafana: http://localhost:3000 (admin/admin)
   - AlertManager: http://localhost:9093

5. Add metrics to your API (if not already present):
   - Use Prometheus Python client:
     pip install prometheus-client
   
   Add to your Flask app:
     from prometheus_client import Counter, Histogram, generate_latest
     
     request_count = Counter('api_requests_total', 'Total requests', ['endpoint', 'method'])
     request_duration = Histogram('api_request_duration_seconds', 'Request duration', ['endpoint'])
     inference_time = Histogram('api_inference_duration_seconds', 'Inference time')
     
     # Use decorators before request/after request

6. Create Grafana dashboards:
   - Import community dashboard or create custom
   - Add panels for key metrics (see list above)
   - Set up alerts with AlertManager

7. Configure notifications:
   - Slack integration
   - Email alerts
   - PagerDuty for critical alerts

8. Test alerts:
   - Stop API container and verify alert fires
   - Generate load and trigger warning alerts
   - Check notification delivery

9. Monitor retention:
   - Prometheus: 30-90 days (configurable)
   - Grafana: Keep dashboards updated
   - AlertManager: Archive fired alerts

10. Scale monitoring:
    - Prometheus Federation for multi-instance
    - Long-term storage (InfluxDB, S3)
    - Distributed tracing (Jaeger, Zipkin)
"""

# ============================================================================
# KEY METRICS TO MONITOR
# ============================================================================

"""
HTTP Request Metrics:
- http_requests_total: Total requests by status code, method, endpoint
- http_request_duration_seconds: Request latency histogram
- http_request_size_bytes: Request/response sizes
- rate(http_requests_total[5m]): Request rate

Model/Inference Metrics:
- inference_duration_seconds: Model inference time
- predictions_total: Total predictions by class
- model_accuracy: Current model accuracy
- confidence_histogram: Distribution of confidence scores

System Metrics:
- container_memory_usage_bytes: Memory usage
- container_cpu_usage_seconds_total: CPU usage
- container_network_transmit_bytes: Network I/O
- up: Service availability

Custom Application Metrics:
- active_requests: Concurrent requests
- queue_depth: Pending requests
- cache_hits: Cache hit rate
- db_query_duration: Database query time (if applicable)

Performance Metrics:
- throughput (predictions/second)
- latency percentiles (p50, p95, p99)
- availability (% uptime)
- error rate (%)
"""

# ============================================================================
# EXAMPLE GRAFANA DASHBOARD QUERIES
# ============================================================================

"""
1. Request Rate (req/s)
   rate(http_requests_total[5m])

2. Error Rate (%)
   (sum(rate(http_requests_total{status=~"5.."}[5m])) /
    sum(rate(http_requests_total[5m]))) * 100

3. Latency (50th, 95th, 99th percentile)
   histogram_quantile(0.50, rate(http_request_duration_seconds_bucket[5m]))
   histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))
   histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))

4. Average Response Time
   avg(http_request_duration_seconds_sum / http_request_duration_seconds_count)

5. Throughput (predictions/sec)
   rate(predictions_total[5m])

6. Average Inference Time
   avg(inference_duration_seconds)

7. Memory Usage (%)
   (container_memory_usage_bytes / container_spec_memory_limit_bytes) * 100

8. CPU Usage (%)
   rate(container_cpu_usage_seconds_total[5m]) * 100

9. Requests by Endpoint
   topk(5, sum by (endpoint) (rate(http_requests_total[5m])))

10. Error Rate by Status Code
    sum by (status) (rate(http_requests_total{status=~"[45].."}[5m]))
"""

# ============================================================================
# TROUBLESHOOTING MONITORING
# ============================================================================

"""
Prometheus not scraping metrics:
1. Check prometheus.yml target configuration
2. Verify API is exposing /metrics endpoint
3. Check firewall/network connectivity
4. Look at Prometheus targets page: http://localhost:9090/targets

Grafana not showing data:
1. Verify Prometheus data source is configured
2. Check data exists in Prometheus
3. Adjust time range (may need historical data)
4. Verify dashboard queries are correct

AlertManager not sending notifications:
1. Check alertmanager.yml configuration
2. Verify webhook/email settings
3. Test with 'amtool' command-line tool
4. Check alertmanager logs: docker logs alertmanager

Missing metrics:
1. Ensure Prometheus client is installed in API
2. Verify metrics are being generated
3. Check scrape interval is not too long
4. Look for permission issues (port access)
"""
