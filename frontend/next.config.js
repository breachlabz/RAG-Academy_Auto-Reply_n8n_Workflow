/** @type {import('next').NextConfig} */
const nextConfig = {
  // Static export: `next build` produces plain HTML/CSS/JS in out/, which
  // api.py serves directly (FileResponse + a StaticFiles mount). No Node
  // server runs at runtime -- this project is one Python container, and
  // adding a second long-running Node process just to host a review page
  // would be a much bigger change than what was asked for.
  output: "export",
  // The page is served at GET /review by FastAPI, so every asset URL Next
  // emits has to carry that prefix too.
  basePath: "/review",
  // The Next.js Image Optimizer needs a running server; static export has
  // none, so images (none in this app yet, but the default would 500) fall
  // back to plain <img> sizing.
  images: { unoptimized: true },
  reactStrictMode: true,
};

module.exports = nextConfig;
