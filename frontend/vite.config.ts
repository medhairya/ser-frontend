import { defineConfig, loadEnv } from "vite";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const api = env.VITE_API_URL || "";
  return {
    define: {
      __API_URL__: JSON.stringify(api.replace(/\/$/, "")),
    },
  };
});
