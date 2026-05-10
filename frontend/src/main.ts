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

function loadGeminiKey(): string {
  try {
    return localStorage.getItem("gemini_api_key") ?? "";
  } catch {
    return "";
  }
}

function saveGeminiKey(key: string): void {
  try {
    localStorage.setItem("gemini_api_key", key.trim());
  } catch {
    /* ignore */
  }
}

async function fetchGeminiInsight(apiKey: string, emotion: string, confidence: number): Promise<string> {
  const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${apiKey}`;
  const prompt = `The user just analyzed a video of themselves speaking. The emotion recognition model detected that they are feeling "${emotion}" with ${(confidence * 100).toFixed(1)}% confidence. Write a short, empathetic, and engaging quote or 2-sentence response directed at the user. If they are happy, share their joy. If they are sad or angry, be supportive and encouraging. Be direct and conversational.`;

  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      contents: [{ parts: [{ text: prompt }] }],
      generationConfig: { temperature: 0.7, maxOutputTokens: 100 }
    })
  });

  if (!response.ok) {
    throw new Error('Gemini API failed');
  }

  const data = await response.json();
  return data.candidates[0].content.parts[0].text;
}

function render(): void {
  const initial = loadStoredApi() || defaultApiUrl();

  document.querySelector<HTMLDivElement>("#app")!.innerHTML = `
    <div class="container">
        <h1>Upload Video for Emotion Recognition</h1>
        
        <form id="uploadForm">
            <div class="api-inputs">
              <input id="apiUrl" name="apiUrl" type="url" autocomplete="off"
                placeholder="Model API URL (e.g. https://...modal.run)"
                value="${initial.replace(/"/g, "&quot;")}" required />
              <input id="geminiKey" name="geminiKey" type="password" autocomplete="off"
                placeholder="Gemini API Key (optional for AI insights)"
                value="${loadGeminiKey().replace(/"/g, "&quot;")}" />
            </div>

            <input type="file" id="videoFile" accept="video/*" required>
            <button type="submit" id="submitBtn">Analyze Emotion</button>
        </form>

        <div class="loader" id="loader"></div>
        <div id="loadingText">Analyzing video and audio... please wait.</div>

        <div id="result"></div>
    </div>
  `;

  const form = document.getElementById('uploadForm') as HTMLFormElement;
  const apiInput = document.getElementById('apiUrl') as HTMLInputElement;
  const geminiInput = document.getElementById('geminiKey') as HTMLInputElement;
  const fileInput = document.getElementById('videoFile') as HTMLInputElement;
  const submitBtn = document.getElementById('submitBtn') as HTMLButtonElement;
  const loader = document.getElementById('loader') as HTMLDivElement;
  const loadingText = document.getElementById('loadingText') as HTMLDivElement;
  const resultDiv = document.getElementById('result') as HTMLDivElement;

  apiInput.addEventListener("change", () => saveApi(apiInput.value));
  apiInput.addEventListener("blur", () => saveApi(apiInput.value));
  geminiInput.addEventListener("change", () => saveGeminiKey(geminiInput.value));
  geminiInput.addEventListener("blur", () => saveGeminiKey(geminiInput.value));

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

  form.addEventListener('submit', async (e) => {
    e.preventDefault();

    if (fileInput.files?.length === 0) {
        alert("Please select a file.");
        return;
    }

    const selected = fileInput.files![0];
    const base = apiInput.value.trim().replace(/\/$/, "");
    const geminiKey = geminiInput.value.trim();
    saveApi(base);
    saveGeminiKey(geminiKey);

    if (!base) {
      alert("Please provide an API URL.");
      return;
    }

    submitBtn.disabled = true;
    loader.style.display = 'block';
    loadingText.style.display = 'block';
    loadingText.innerText = "Analyzing video and audio... please wait.";
    loadingText.style.color = "#cbd5e1";
    resultDiv.innerHTML = '';
    
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

      resultDiv.innerHTML = `
        <div class="result-text">${data.emotion.toUpperCase()}</div>
        <div class="result-conf">Confidence: ${(data.confidence * 100).toFixed(1)}%</div>
        <div class="bars">${renderBars(data.probabilities)}</div>
        <div id="geminiResponse" class="gemini-card" style="display: none;"></div>
      `;

      if (geminiKey) {
        const geminiDiv = document.getElementById('geminiResponse') as HTMLDivElement;
        geminiDiv.style.display = 'block';
        geminiDiv.innerHTML = `<div class="loader-small"></div> Fetching AI insight...`;
        
        try {
          const insight = await fetchGeminiInsight(geminiKey, data.emotion, data.confidence);
          geminiDiv.innerHTML = `<strong>✨ AI Insight:</strong><br/><br/>${insight}`;
        } catch (err) {
          console.error(err);
          geminiDiv.innerHTML = `<em>Could not fetch AI insight. Please check your Gemini API key.</em>`;
        }
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      loadingText.innerText = `Error: ${msg}`;
      loadingText.style.color = "#ef4444";
    } finally {
      submitBtn.disabled = false;
      loader.style.display = 'none';
      if (resultDiv.innerHTML !== '') {
          loadingText.style.display = 'none';
      }
    }
  });
}

render();
