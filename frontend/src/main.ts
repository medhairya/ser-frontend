import "./style.css";

type PredictSuccess = {
  status: "success";
  emotion: string;
  confidence: number;
  probabilities: Record<string, number>;
};

type PredictErrorBody = { detail?: string };

function isPredictSuccess(data: unknown): data is PredictSuccess {
  if (!data || typeof data !== "object") return false;
  const o = data as Record<string, unknown>;
  return (
    o.status === "success" &&
    typeof o.emotion === "string" &&
    typeof o.confidence === "number" &&
    o.probabilities !== null &&
    typeof o.probabilities === "object"
  );
}

const ORDER = [
  "neutral",
  "calm",
  "happy",
  "sad",
  "angry",
  "fearful",
  "disgust",
  "surprised",
] as const;

function defaultApiUrl(): string {
  const fromDefine = typeof __API_URL__ !== "undefined" ? __API_URL__ : "";
  const fromImport = import.meta.env.VITE_API_URL ?? "";
  return (fromDefine || fromImport).replace(/\/$/, "");
}

function loadStoredApi(): string {
  try {
    return localStorage.getItem("ser_api_url") ?? "";
  } catch {
    return "";
  }
}

function saveApi(url: string): void {
  try {
    localStorage.setItem("ser_api_url", url.trim());
  } catch {
    /* ignore */
  }
}

function render(): void {
  const initial = loadStoredApi() || defaultApiUrl();

  document.querySelector<HTMLDivElement>("#app")!.innerHTML = `
    <header>
      <p class="eyebrow">Speech & video · RAVDESS-style clip</p>
      <h1>Multimodal emotion recognition</h1>
      <p class="lead">
        Upload a short talking-head video. The AVT-CA model fuses mel-spectrogram audio with
        sampled RGB frames—same setup as the RAVDESS training pipeline.
      </p>
    </header>

    <section class="panel" aria-label="Upload and analyze">
      <div class="api-row">
        <label for="apiUrl">API base URL</label>
        <input id="apiUrl" name="apiUrl" type="url" autocomplete="off"
          placeholder="https://YOUR_WORKSPACE--ser-avtca-api-serve.modal.run"
          value="${initial.replace(/"/g, "&quot;")}" />
      </div>
      <p class="hint">
        Set <code style="font-family:var(--mono)">VITE_API_URL</code> in Vercel for a default, or paste your Modal URL here (saved in the browser).
      </p>

      <label class="dropzone" id="dropzone" tabindex="0">
        <input id="file" type="file" accept="video/mp4,video/webm,video/quicktime,video/x-msvideo,.mp4,.webm,.mov,.avi,.mkv" />
        <strong>Choose a video</strong> or drag it here
        <p>MP4 / WebM / MOV · includes audio track recommended</p>
      </label>

      <div class="actions">
        <button type="button" class="primary" id="run" disabled>Run inference</button>
        <span class="hint" id="fileName">No file selected</span>
      </div>

      <div class="status" id="status" role="status"></div>
      <div id="out" class="result" hidden></div>
    </section>

    <footer>
      Paper: Venkatraman et al., “Multimodal Emotion Recognition using Audio-Video Transformer Fusion with Cross Attention” (arXiv:2407.18552).
      Train with <code style="font-family:var(--mono)">deploy/train.py</code>; host the GPU API on Modal.
    </footer>
  `;

  const apiInput = document.querySelector<HTMLInputElement>("#apiUrl")!;
  const fileInput = document.querySelector<HTMLInputElement>("#file")!;
  const dropzone = document.querySelector<HTMLLabelElement>("#dropzone")!;
  const runBtn = document.querySelector<HTMLButtonElement>("#run")!;
  const fileName = document.querySelector<HTMLSpanElement>("#fileName")!;
  const statusEl = document.querySelector<HTMLDivElement>("#status")!;
  const outEl = document.querySelector<HTMLDivElement>("#out")!;

  let selected: File | null = null;

  apiInput.addEventListener("change", () => saveApi(apiInput.value));
  apiInput.addEventListener("blur", () => saveApi(apiInput.value));

  const setFile = (file: File | null) => {
    selected = file;
    runBtn.disabled = !selected;
    fileName.textContent = selected ? selected.name : "No file selected";
    outEl.hidden = true;
    outEl.innerHTML = "";
    statusEl.textContent = "";
    statusEl.className = "status";
  };

  fileInput.addEventListener("change", () => {
    const f = fileInput.files?.[0] ?? null;
    setFile(f);
  });

  ["dragenter", "dragover"].forEach((ev) => {
    dropzone.addEventListener(ev, (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });
  });
  ["dragleave", "drop"].forEach((ev) => {
    dropzone.addEventListener(ev, (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
    });
  });
  dropzone.addEventListener("drop", (e) => {
    const f = e.dataTransfer?.files?.[0];
    if (f) {
      fileInput.files = e.dataTransfer!.files;
      setFile(f);
    }
  });
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });

  const renderBars = (probs: Record<string, number>) => {
    const keys = ORDER.filter((k) => k in probs);
    const extras = Object.keys(probs).filter((k) => !keys.includes(k as (typeof ORDER)[number]));
    const ordered = [...keys, ...extras];
    return ordered
      .map((name) => {
        const v = Math.round((probs[name] ?? 0) * 1000) / 10;
        return `
        <div class="bar-row">
          <span class="bar-name">${name}</span>
          <div class="bar-track"><div class="bar-fill" style="width:${v}%"></div></div>
          <span class="bar-pct">${v}%</span>
        </div>`;
      })
      .join("");
  };

  runBtn.addEventListener("click", async () => {
    if (!selected) return;

    const base = apiInput.value.trim().replace(/\/$/, "");
    saveApi(base);
    if (!base) {
      statusEl.textContent = "Set the API base URL to your Modal deployment.";
      statusEl.className = "status error";
      return;
    }

    statusEl.innerHTML = `<span class="loader"><span class="spinner" aria-hidden="true"></span> Uploading and running GPU inference…</span>`;
    statusEl.className = "status";
    outEl.hidden = true;
    runBtn.disabled = true;

    const fd = new FormData();
    fd.append("file", selected, selected.name);

    try {
      const res = await fetch(`${base}/predict`, {
        method: "POST",
        body: fd,
      });
      const data: unknown = await res.json();

      if (!res.ok) {
        const body = data as PredictErrorBody;
        const msg = body?.detail != null ? String(body.detail) : res.statusText;
        throw new Error(msg || `HTTP ${res.status}`);
      }

      if (!isPredictSuccess(data)) {
        throw new Error("Unexpected response from API");
      }

      statusEl.textContent = "Done.";
      statusEl.className = "status ok";
      outEl.hidden = false;
      outEl.innerHTML = `
        <h2>Prediction</h2>
        <div class="prediction">
          <span class="label">${data.emotion}</span>
          <span class="conf">confidence ${(data.confidence * 100).toFixed(1)}%</span>
        </div>
        <div class="bars">${renderBars(data.probabilities)}</div>
      `;
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      statusEl.textContent = `Request failed: ${msg}`;
      statusEl.className = "status error";
    } finally {
      runBtn.disabled = !selected;
    }
  });
}

render();
