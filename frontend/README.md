# VacancyLens frontend

See the [project README](../README.md) for complete setup, model training, results and verification.

From the project root:

```bash
npm --prefix frontend ci
npm --prefix frontend run dev
```

The FastAPI server must run on 127.0.0.1:8000. Open http://localhost:5173. Relative `/api` URLs use Vite's development proxy. `npm --prefix frontend run build` includes TypeScript checking. Browser integration tests are documented in the root README.
