# CopilotKit AG-UI Runtime — Self-hosted bridge for agentic dashboard
#
# This container runs a lightweight Node.js server that bridges the
# CopilotKit React frontend SDK to Pydantic AI agents via the AG-UI
# streaming protocol. It proxies AG-UI events between the React
# dashboard-ui and the Python dashboard-api (FastAPI + AGUIAdapter).
#
# Architecture:
#   dashboard-ui (React) → copilotkit (this) → dashboard-api (FastAPI/AG-UI)
#
# NOTE: This is a placeholder runtime. The actual CopilotKit runtime
# (@copilotkit/runtime) is a Node.js package. For now, we use a thin
# nginx reverse proxy that forwards AG-UI requests to the dashboard-api
# service, which implements the AG-UI protocol directly via Pydantic AI.
#
# When CopilotKit publishes an official Docker image, replace this with:
#   image: copilotkit/runtime:latest

FROM node:20-alpine AS runtime

LABEL project="minivess-mlops"
LABEL service="copilotkit"
LABEL description="CopilotKit AG-UI runtime bridge for agentic dashboard"

RUN addgroup -g 1000 copilotkit && \
    adduser -u 1000 -G copilotkit -D copilotkit

WORKDIR /app

# Minimal AG-UI proxy server — forwards requests to dashboard-api
COPY deployment/copilotkit/server.mjs /app/server.mjs

USER copilotkit

ENV COPILOTKIT_PORT=4111
ENV DASHBOARD_API_URL=http://dashboard-api:8090

EXPOSE 4111

CMD ["node", "/app/server.mjs"]
