# CloudFront + S3 (frontend)

- Build the SPA with `npm run build` in `frontend/`, upload `dist/` to the S3 bucket used as origin.
- Default root object: `index.html`. Add a custom error response for `403`/`404` to `/index.html` so client-side routes work.
- Invalidate CloudFront after deploy when using versioned assets or when troubleshooting caching issues.
