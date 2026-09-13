/**
 * ZOOVOX – Real Backend API Client
 * All mock data removed. Every call goes to the FastAPI backend.
 *
 * Backend runs at: http://localhost:8000  (dev)
 * Change VITE_API_URL in .env for production.
 */

export const BASE_URL =
  (import.meta.env.VITE_API_URL as string) ?? "http://localhost:8000/api/v1";

// ── Token helpers ─────────────────────────────────────────────────────────────
export const token = {
  get: () => localStorage.getItem("zv_access"),
  getRefresh: () => localStorage.getItem("zv_refresh"),
  save: (access: string, refresh: string) => {
    localStorage.setItem("zv_access", access);
    localStorage.setItem("zv_refresh", refresh);
  },
  clear: () => {
    localStorage.removeItem("zv_access");
    localStorage.removeItem("zv_refresh");
    localStorage.removeItem("zv_user");
  },
  saveUser: (u: User) => localStorage.setItem("zv_user", JSON.stringify(u)),
  getUser: (): User | null => {
    const s = localStorage.getItem("zv_user");
    return s ? (JSON.parse(s) as User) : null;
  },
};

// ── Base fetch wrapper ────────────────────────────────────────────────────────
async function request<T>(
  method: string,
  path: string,
  body?: unknown,
  isForm = false
): Promise<T> {
  const headers: Record<string, string> = {};
  const t = token.get();
  if (t) headers["Authorization"] = `Bearer ${t}`;
  if (!isForm) headers["Content-Type"] = "application/json";

  const res = await fetch(`${BASE_URL}${path}`, {
    method,
    headers,
    body: isForm
      ? (body as FormData)
      : body !== undefined
      ? JSON.stringify(body)
      : undefined,
  });

  if (res.status === 401) {
    token.clear();
    window.location.href = "/";
    throw new Error("Session expired");
  }

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Request failed" }));
    throw new Error(err.detail ?? "Request failed");
  }

  return res.json() as Promise<T>;
}

// ── Types ─────────────────────────────────────────────────────────────────────
export interface User {
  id: string;
  name: string;
  email: string;
  face_enrolled: boolean;
  plan: "free" | "pro" | "enterprise";
  animals_analyzed: number;
  created_at: string;
}

export interface AuthResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
  user: User;
}

export interface AudioAnalysisResult {
  session_id: string;
  animal_type: string;
  animal_confidence: number;
  detected_emotion: string;
  emotion_confidence: number;
  translation_en: string;
  audio_duration_sec: number;
  behavioral_context: string;
  research_reference: string;
  processing_time_ms: number;
  raw_yamnet_scores: Record<string, number>;
  // Which model produced animal_confidence: a real trained model
  // ("zoovox_classifier" | "yamnet") or a rule-based estimate ("heuristic").
  // Optional because older cached responses may predate this field.
  prediction_source?: "zoovox_classifier" | "yamnet" | "heuristic";
}

export interface InterimAudioAnalysis {
  type: "interim";
  animal_type: string;
  confidence: number;
  emotion: string;
}

export interface AudioStreamCallbacks {
  onInterim?: (result: InterimAudioAnalysis) => void;
  onFinal?: (result: AudioAnalysisResult) => void;
  onError?: (message: string) => void;
}

export interface AudioStream {
  ready: Promise<void>;
  sendChunk: (chunk: Blob | ArrayBuffer) => void;
  end: () => void;
  close: () => void;
}

export interface HumanToAnimalResult {
  session_id: string;
  original_text: string;
  animal_cue_description: string;
  audio_url: string;
  recommended_actions: string[];
  scientific_basis: string;
}

export interface VetService {
  id: string;
  name: string;
  type: string;
  address: string;
  distance_km: number;
  latitude: number;
  longitude: number;
  phone?: string;
  website?: string;
  rating?: number;
  open_now?: boolean;
  hours?: string;
  google_place_id?: string;
}

export interface HistoryItem {
  session_id: string;
  direction: "animal_to_human" | "human_to_animal";
  animal_type: string;
  translation: string;
  confidence: number;
  created_at: string;
}

export interface DashboardStats {
  total_sessions: number;
  animals_analyzed: number;
  animal_breakdown: Record<string, number>;
  emotion_breakdown: Record<string, number>;
  avg_confidence: number;
  top_animals: string[];
  weekly_trend: { date: string; count: number }[];
  plan: string;
}

export interface SupportedAnimalsResponse {
  animals: { id: string; name: string; emoji: string; research: string }[];
}

// Species/taxonomy reference lookup (GBIF) — scientific classification only.
// Unrelated to, and never a validation of, animal-sound/vocalization
// classification results shown elsewhere in the app.
export interface TaxonomyClassification {
  kingdom: string | null;
  phylum: string | null;
  class_name: string | null;
  order: string | null;
  family: string | null;
  genus: string | null;
  species: string | null;
}

export interface TaxonomyResponse {
  query: string;
  found: boolean;
  taxon_id: string | null;
  scientific_name: string | null;
  canonical_name: string | null;
  common_names: string[];
  rank: string | null;
  match_type: string | null;
  classification: TaxonomyClassification | null;
  match_confidence: number | null;
  source: string;
  source_url: string | null;
  attribution: string | null;
}

function getAudioStreamUrl(): string {
  const configured = import.meta.env.VITE_WS_URL as string | undefined;
  if (configured) {
    return `${configured.replace(/\/$/, "").replace(/\/api\/v1$/, "")}/api/v1/audio/stream`;
  }

  const apiOrigin = BASE_URL.replace(/\/api\/v1\/?$/, "");
  return `${apiOrigin.replace(/^http:/, "ws:").replace(/^https:/, "wss:")}/api/v1/audio/stream`;
}

// ── Auth API ──────────────────────────────────────────────────────────────────
export const authApi = {
  async register(name: string, email: string, password: string): Promise<AuthResponse> {
    const data = await request<AuthResponse>("POST", "/auth/register", {
      name,
      email,
      password,
    });
    token.save(data.access_token, data.refresh_token);
    token.saveUser(data.user);
    return data;
  },

  async login(email: string, password: string): Promise<AuthResponse> {
    const data = await request<AuthResponse>("POST", "/auth/login", {
      email,
      password,
    });
    token.save(data.access_token, data.refresh_token);
    token.saveUser(data.user);
    return data;
  },

  async faceLogin(frameB64: string): Promise<AuthResponse> {
    const data = await request<AuthResponse>("POST", "/auth/face/login", {
      frame_b64: frameB64,
    });
    token.save(data.access_token, data.refresh_token);
    token.saveUser(data.user);
    return data;
  },

  async enrollFace(frameB64: string): Promise<{ success: boolean; message: string }> {
    return request("POST", "/auth/face/enroll", {
      frame_b64: frameB64,
      user_id: "",
    });
  },

  getMe(): Promise<User> {
    return request("GET", "/auth/me");
  },

  logout() {
    token.clear();
  },
};

// ── Audio API ─────────────────────────────────────────────────────────────────
export const audioApi = {
  async analyzeAudio(
    audioBlob: Blob,
    language = "en"
  ): Promise<AudioAnalysisResult> {
    const fd = new FormData();
    fd.append("audio", audioBlob, "recording.webm");
    fd.append("language", language);
    return request("POST", "/audio/analyze", fd, true);
  },

  async humanToAnimal(
    text: string,
    targetAnimal: string,
    language = "en"
  ): Promise<HumanToAnimalResult> {
    return request("POST", "/audio/human-to-animal", {
      text,
      target_animal: targetAnimal,
      language,
    });
  },

  async getHistory(
    page = 1,
    limit = 20
  ): Promise<{ items: HistoryItem[]; total: number; pages: number }> {
    return request("GET", `/audio/history?page=${page}&limit=${limit}`);
  },

  async getSupportedAnimals(): Promise<{
    animals: { id: string; name: string; emoji: string; research: string }[];
  }> {
    return request("GET", "/audio/supported-animals");
  },

  /**
   * Opens the README-defined authenticated WebSocket stream. Send each
   * MediaRecorder chunk as it arrives; the API emits interim predictions and
   * one persisted final analysis after `end()`.
   */
  createAudioStream(
    callbacks: AudioStreamCallbacks,
    options: { contentType?: string; language?: string } = {}
  ): AudioStream {
    const accessToken = token.get();
    let settled = false;
    let ended = false;
    let finalReceived = false;
    let clientClosed = false;
    let failed = false;
    let resolveReady: () => void;
    let rejectReady: (reason?: unknown) => void;
    const ready = new Promise<void>((resolve, reject) => {
      resolveReady = resolve;
      rejectReady = reject;
    });

    if (!accessToken) {
      const error = "Sign in is required before starting a live audio stream.";
      callbacks.onError?.(error);
      rejectReady!(new Error(error));
      return {
        ready,
        sendChunk: () => undefined,
        end: () => undefined,
        close: () => undefined,
      };
    }

    const ws = new WebSocket(getAudioStreamUrl());

    const fail = (message: string) => {
      if (failed || finalReceived || clientClosed) return;
      failed = true;
      callbacks.onError?.(message);
      if (!settled) {
        settled = true;
        rejectReady!(new Error(message));
      }
    };

    ws.onopen = () => {
      ws.send(
        JSON.stringify({
          type: "auth",
          token: accessToken,
          content_type: options.contentType ?? "audio/webm",
          language: options.language ?? "en",
        })
      );
    };

    ws.onmessage = (event) => {
      let message: { type?: string; message?: string } & Record<string, unknown>;
      try {
        message = JSON.parse(event.data as string);
      } catch {
        fail("The live-audio server returned an invalid response.");
        return;
      }

      if (message.type === "auth_ok") {
        if (!settled) {
          settled = true;
          resolveReady!();
        }
      } else if (message.type === "interim") {
        callbacks.onInterim?.(message as unknown as InterimAudioAnalysis);
      } else if (message.type === "final") {
        finalReceived = true;
        callbacks.onFinal?.(message as unknown as AudioAnalysisResult);
      } else if (message.type === "error") {
        fail(message.message ?? "Live audio analysis failed.");
      }
    };

    ws.onerror = () => fail("Unable to connect to the live-audio service.");
    ws.onclose = () => {
      if (!finalReceived && !clientClosed) fail("The live-audio connection closed before analysis completed.");
    };

    return {
      ready,
      sendChunk: (chunk) => {
        if (!ended && ws.readyState === WebSocket.OPEN) ws.send(chunk);
      },
      end: () => {
        if (!ended && ws.readyState === WebSocket.OPEN) {
          ended = true;
          ws.send(JSON.stringify({ type: "end" }));
        }
      },
      close: () => {
        clientClosed = true;
        ended = true;
        ws.close();
      },
    };
  },
};

// ── Veterinary API ────────────────────────────────────────────────────────────
export const vetApi = {
  async searchNearby(
    lat: number,
    lng: number,
    radiusKm = 10,
    type = "hospital"
  ): Promise<{ results: VetService[]; total: number }> {
    return request(
      "GET",
      `/vet/search?latitude=${lat}&longitude=${lng}&radius_km=${radiusKm}&type=${type}`
    );
  },

  async getPetCareTips(animal: string): Promise<{
    animal: string;
    tips: Record<string, string[]>;
  }> {
    return request("GET", `/vet/pet-care-tips/${animal}`);
  },
};

// ── Analytics API ─────────────────────────────────────────────────────────────
export const analyticsApi = {
  getDashboard(): Promise<DashboardStats> {
    return request("GET", "/analytics/dashboard");
  },
};

// ── Species/Taxonomy API (GBIF reference lookup) ─────────────────────────────
// Scope: scientific classification only. Never treat this as, or present it
// alongside, animal-sound/vocalization classification confidence — those are
// a separate, unrelated result.
export const taxonomyApi = {
  lookupSpecies(commonName: string): Promise<TaxonomyResponse> {
    return request("GET", `/species/${encodeURIComponent(commonName)}`);
  },
};

// ── Capture webcam frame as base64 JPEG ───────────────────────────────────────
export function captureFrame(video: HTMLVideoElement): string {
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  const ctx = canvas.getContext("2d")!;
  ctx.drawImage(video, 0, 0);
  // Return base64 without the data:image/jpeg;base64, prefix
  return canvas.toDataURL("image/jpeg", 0.85).split(",")[1];
}