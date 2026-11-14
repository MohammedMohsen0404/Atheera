import type { NextConfig } from "next";

// The dashboard is exported as static files into ./dist, which the FastAPI backend serves.
// During `next dev`, /api is proxied to the backend; the telemetry WebSocket connects to :8010 directly.
const backend = process.env.ATHEER_BACKEND_URL ?? "http://127.0.0.1:8010";

const nextConfig: NextConfig = {
  output: "export",
  distDir: "dist",
  trailingSlash: true,
  poweredByHeader: false,
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
  ...(process.env.NODE_ENV === "development"
    ? { async rewrites() { return [{ source: "/api/:path*", destination: `${backend}/api/:path*` }]; } }
    : {}),
};

export default nextConfig;
