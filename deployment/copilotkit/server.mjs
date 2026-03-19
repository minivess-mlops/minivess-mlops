/**
 * CopilotKit AG-UI Runtime Proxy
 *
 * Minimal Node.js server that proxies AG-UI protocol requests from the
 * CopilotKit React frontend to the Pydantic AI dashboard-api backend.
 *
 * This is a thin bridge — the actual AG-UI protocol handling happens
 * in the Python dashboard-api service via pydantic_ai.ui.ag_ui.AGUIAdapter.
 *
 * Environment variables:
 *   COPILOTKIT_PORT     — Port to listen on (default: 4111)
 *   DASHBOARD_API_URL   — Backend AG-UI endpoint (default: http://dashboard-api:8090)
 */

import http from "node:http";

const PORT = parseInt(process.env.COPILOTKIT_PORT || "4111", 10);
const BACKEND_URL = process.env.DASHBOARD_API_URL || "http://dashboard-api:8090";

const server = http.createServer((req, res) => {
  // Health check
  if (req.url === "/health" && req.method === "GET") {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ status: "ok", service: "copilotkit-runtime" }));
    return;
  }

  // CORS headers for AG-UI streaming
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, GET, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, Accept");

  if (req.method === "OPTIONS") {
    res.writeHead(204);
    res.end();
    return;
  }

  // Proxy AG-UI requests to dashboard-api
  if (req.method === "POST") {
    const targetUrl = new URL(req.url || "/ag-ui", BACKEND_URL);

    let body = [];
    req.on("data", (chunk) => body.push(chunk));
    req.on("end", () => {
      const payload = Buffer.concat(body);

      const proxyReq = http.request(
        targetUrl,
        {
          method: "POST",
          headers: {
            "Content-Type": req.headers["content-type"] || "application/json",
            Accept: req.headers["accept"] || "text/event-stream",
            "Content-Length": payload.length,
          },
        },
        (proxyRes) => {
          res.writeHead(proxyRes.statusCode || 200, proxyRes.headers);
          proxyRes.pipe(res);
        }
      );

      proxyReq.on("error", (err) => {
        console.error("Proxy error:", err.message);
        res.writeHead(502, { "Content-Type": "application/json" });
        res.end(
          JSON.stringify({
            error: "Backend unavailable",
            detail: err.message,
          })
        );
      });

      proxyReq.write(payload);
      proxyReq.end();
    });
    return;
  }

  res.writeHead(404, { "Content-Type": "application/json" });
  res.end(JSON.stringify({ error: "Not found" }));
});

server.listen(PORT, "0.0.0.0", () => {
  console.log(`CopilotKit AG-UI proxy listening on port ${PORT}`);
  console.log(`Proxying to: ${BACKEND_URL}`);
});
