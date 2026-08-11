/**
 * ZOOVOX Frontend ↔ Backend Integration
 * ════════════════════════════════════════
 * Drop this file into: src/lib/api.ts
 *
 * Replace the mock handlers in Auth.tsx, Dashboard.tsx, Veterinary.tsx
 * with these real API calls.
 *
 * Usage:
 *   import { api } from '@/lib/api'
 *   const result = await api.auth.login(email, password)
 */

const BASE_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000/api/v1'

// ── Token management ────────────────────────────────────────────────────────

function getToken(): string | null {
  return localStorage.getItem('zoovox_access_token')
}

function saveTokens(access: string, refresh: string) {
  localStorage.setItem('zoovox_access_token', access)
  localStorage.setItem('zoovox_refresh_token', refresh)
}

function clearTokens() {
  localStorage.removeItem('zoovox_access_token')
  localStorage.removeItem('zoovox_refresh_token')
}

// ── Base fetcher ─────────────────────────────────────────────────────────────

async function req<T>(
  method: string,
  path: string,
  body?: unknown,
  isFormData = false,
): Promise<T> {
  const token = getToken()
  const headers: Record<string, string> = {}
  if (token) headers['Authorization'] = `Bearer ${token}`
  if (!isFormData) headers['Content-Type'] = 'application/json'

  const res = await fetch(`${BASE_URL}${path}`, {
    method,
    headers,
    body: isFormData ? (body as FormData) : body ? JSON.stringify(body) : undefined,
  })

  if (res.status === 401) {
    clearTokens()
    window.location.href = '/'
    throw new Error('Session expired')
  }

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Unknown error' }))
    throw new Error(err.detail ?? 'Request failed')
  }

  return res.json() as Promise<T>
}

// ── Auth API ─────────────────────────────────────────────────────────────────

export const authApi = {

  async register(name: string, email: string, password: string) {
    const data = await req<{ access_token: string; refresh_token: string; user: User }>(
      'POST', '/auth/register', { name, email, password }
    )
    saveTokens(data.access_token, data.refresh_token)
    return data
  },

  async login(email: string, password: string) {
    const data = await req<{ access_token: string; refresh_token: string; user: User }>(
      'POST', '/auth/login', { email, password }
    )
    saveTokens(data.access_token, data.refresh_token)
    return data
  },

  async enrollFace(frameB64: string) {
    const token = getToken()
    return req('POST', '/auth/face/enroll', {
      frame_b64: frameB64,
      user_id: '',   // backend reads from JWT
    })
  },

  async faceLogin(frameB64: string) {
    const data = await req<{ access_token: string; refresh_token: string; user: User }>(
      'POST', '/auth/face/login', { frame_b64: frameB64 }
    )
    saveTokens(data.access_token, data.refresh_token)
    return data
  },

  logout() {
    clearTokens()
  },

  getMe(): Promise<User> {
    return req('GET', '/auth/me')
  },
}

// ── Audio API ─────────────────────────────────────────────────────────────────

export const audioApi = {

  async analyzeAudio(audioBlob: Blob, language = 'en'): Promise<AudioAnalysisResult> {
    const fd = new FormData()
    fd.append('audio', audioBlob, 'recording.webm')
    fd.append('language', language)
    return req('POST', '/audio/analyze', fd, true)
  },

  async humanToAnimal(text: string, targetAnimal: string, language = 'en') {
    return req('POST', '/audio/human-to-animal', { text, target_animal: targetAnimal, language })
  },

  getHistory(page = 1, limit = 20, animal?: string) {
    const params = new URLSearchParams({ page: String(page), limit: String(limit) })
    if (animal) params.append('animal', animal)
    return req<{ items: TranslationHistoryItem[]; total: number }>('GET', `/audio/history?${params}`)
  },

  getSupportedAnimals() {
    return req<{ animals: Animal[] }>('GET', '/audio/supported-animals')
  },

  // ── WebSocket streaming ───────────────────────────────────────────────────

  createAudioStream(onInterim: (data: unknown) => void, onFinal: (data: AudioAnalysisResult) => void) {
    const WS_URL = (import.meta.env.VITE_WS_URL ?? 'ws://localhost:8000') + '/api/v1/audio/stream'
    const ws = new WebSocket(WS_URL)
    const token = getToken()

    ws.onopen = () => {
      ws.send(JSON.stringify({ type: 'auth', token }))
    }

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data)
      if (msg.type === 'interim') onInterim(msg)
      if (msg.type === 'final') onFinal(msg)
    }

    const sendChunk = (chunk: ArrayBuffer) => {
      const b64 = btoa(String.fromCharCode(...new Uint8Array(chunk)))
      ws.send(JSON.stringify({ type: 'audio_chunk', data: b64 }))
    }

    const end = () => {
      ws.send(JSON.stringify({ type: 'end' }))
    }

    return { ws, sendChunk, end }
  }
}

// ── Veterinary API ─────────────────────────────────────────────────────────

export const vetApi = {
  searchNearby(lat: number, lng: number, radiusKm = 10, type = 'hospital') {
    return req<{ results: VetService[]; total: number }>(
      'GET', `/vet/search?latitude=${lat}&longitude=${lng}&radius_km=${radiusKm}&type=${type}`
    )
  },
  getPetCareTips(animal: string) {
    return req('GET', `/vet/pet-care-tips/${animal}`)
  },
}

// ── Analytics API ──────────────────────────────────────────────────────────

export const analyticsApi = {
  getDashboard() {
    return req('GET', '/analytics/dashboard')
  },
  getLeaderboard() {
    return req('GET', '/analytics/leaderboard')
  },
}

// ── Types ──────────────────────────────────────────────────────────────────

export interface User {
  id: string
  name: string
  email: string
  face_enrolled: boolean
  plan: 'free' | 'pro' | 'enterprise'
  animals_analyzed: number
  created_at: string
}

export interface AudioAnalysisResult {
  session_id: string
  animal_type: string
  animal_confidence: number
  detected_emotion: string
  emotion_confidence: number
  translation_en: string
  audio_duration_sec: number
  behavioral_context: string
  research_reference: string
  processing_time_ms: number
  raw_yamnet_scores: Record<string, number>
}

export interface TranslationHistoryItem {
  session_id: string
  direction: 'animal_to_human' | 'human_to_animal'
  animal_type: string
  translation: string
  confidence: number
  created_at: string
}

export interface VetService {
  id: string
  name: string
  type: string
  address: string
  distance_km: number
  latitude: number
  longitude: number
  phone?: string
  rating?: number
  open_now?: boolean
  hours?: string
}

export interface Animal {
  id: string
  name: string
  emoji: string
  emotions: number
  research: string
}

// ── Default export ─────────────────────────────────────────────────────────

export const api = { auth: authApi, audio: audioApi, vet: vetApi, analytics: analyticsApi }
export default api
