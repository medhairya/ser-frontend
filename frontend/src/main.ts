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

/** Production Hugging Face Space (fixed; not user-editable). */
const API_BASE = "https://dhairya-4252-dhairya-hf-space.hf.space".replace(/\/$/, "");

function render(): void {
  document.querySelector<HTMLDivElement>("#app")!.innerHTML = `
    <div class="container">
        <h1>Upload Video for Emotion Recognition</h1>
        <p class="api-endpoint">API: ${API_BASE.replace(/</g, "&lt;").replace(/&/g, "&amp;")}</p>

        <form id="uploadForm">
            <input type="file" id="videoFile" accept="video/*" required>
            <button type="submit" id="submitBtn">Analyze Emotion</button>
        </form>

        <div class="loader" id="loader"></div>
        <div id="loadingText">Analyzing video and audio... please wait.</div>

        <div id="result"></div>
    </div>
  `;

  const form = document.getElementById("uploadForm") as HTMLFormElement;
  const fileInput = document.getElementById("videoFile") as HTMLInputElement;
  const submitBtn = document.getElementById("submitBtn") as HTMLButtonElement;
  const loader = document.getElementById("loader") as HTMLDivElement;
  const loadingText = document.getElementById("loadingText") as HTMLDivElement;
  const resultDiv = document.getElementById("result") as HTMLDivElement;

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

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    if (fileInput.files?.length === 0) {
      alert("Please select a file.");
      return;
    }

    const selected = fileInput.files![0];

    submitBtn.disabled = true;
    loader.style.display = "block";
    loadingText.style.display = "block";
    loadingText.innerText = "Analyzing video and audio... please wait.";
    loadingText.style.color = "#cbd5e1";
    resultDiv.innerHTML = "";

    const fd = new FormData();
    fd.append("file", selected, selected.name);

    try {
      const res = await fetch(`${API_BASE}/predict`, {
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

      resultDiv.innerHTML = `
        <div class="result-text">${data.emotion.toUpperCase()}</div>
        <div class="result-conf">Confidence: ${(data.confidence * 100).toFixed(1)}%</div>
        <div class="bars">${renderBars(data.probabilities)}</div>
      `;
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      loadingText.innerText = `Error: ${msg}`;
      loadingText.style.color = "#ef4444";
    } finally {
      submitBtn.disabled = false;
      loader.style.display = "none";
      if (resultDiv.innerHTML !== "") {
        loadingText.style.display = "none";
      }
    }
  });
}

render();
