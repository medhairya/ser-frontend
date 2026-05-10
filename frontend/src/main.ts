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

// Replace this placeholder with your actual Gemini API Key
const GEMINI_API_KEY = "AIzaSyAEZHyfa6NRk3RRVcAr2G2uK3NEnlq9goQ";

async function fetchGeminiInsight(emotion: string, confidence: number): Promise<string> {
  const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${GEMINI_API_KEY}`;
  const prompt = `You are an empathetic and insightful AI assistant. The user just analyzed a video of themselves speaking. The emotion recognition model detected that they are feeling "${emotion}" with ${(confidence * 100).toFixed(1)}% confidence. 
Please write a highly engaging, meaningful, and thoughtful response consisting of at least 3 to 4 full sentences directed at the user. 
If they are happy, calm, or neutral, share their joy and provide an inspiring quote. 
If they are sad, fearful, surprised, disgusted, or angry, be highly supportive, encouraging, and provide a comforting thought. 
Do not be brief. Be conversational, thoughtful, and expressive.`;
  
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      contents: [{ parts: [{ text: prompt }] }],
      generationConfig: { temperature: 0.7 }
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
  const fileInput = document.getElementById('videoFile') as HTMLInputElement;
  const submitBtn = document.getElementById('submitBtn') as HTMLButtonElement;
  const loader = document.getElementById('loader') as HTMLDivElement;
  const loadingText = document.getElementById('loadingText') as HTMLDivElement;
  const resultDiv = document.getElementById('result') as HTMLDivElement;

  apiInput.addEventListener("change", () => saveApi(apiInput.value));
  apiInput.addEventListener("blur", () => saveApi(apiInput.value));


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
    saveApi(base);

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

      if (GEMINI_API_KEY && GEMINI_API_KEY !== "AIzaSyAEZHyfa6NRk3RRVcAr2G2uK3NEnlq9goQ") {
        const geminiDiv = document.getElementById('geminiResponse') as HTMLDivElement;
        geminiDiv.style.display = 'block';
        geminiDiv.innerHTML = `<div class="loader-small"></div> Fetching AI insight...`;
        
        try {
          const insight = await fetchGeminiInsight(data.emotion, data.confidence);
          geminiDiv.innerHTML = `<strong>✨ AI Insight:</strong><br/><br/>${insight}`;
        } catch (err) {
          console.error(err);
          geminiDiv.innerHTML = `<em>Could not fetch AI insight. Please check your hardcoded Gemini API key.</em>`;
        }
      } else {
        const geminiDiv = document.getElementById('geminiResponse') as HTMLDivElement;
        geminiDiv.style.display = 'block';
        geminiDiv.innerHTML = `<em>Please set your GEMINI_API_KEY in main.ts to see AI insights.</em>`;
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
