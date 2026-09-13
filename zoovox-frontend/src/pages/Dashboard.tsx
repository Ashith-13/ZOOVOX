import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { BookOpen, Clock, LogOut, MapPin, Menu, Mic, PawPrint, Volume2, X } from "lucide-react";
import { toast } from "sonner";
import { useNavigate } from "react-router-dom";
import voiceWaves from "@/assets/voice-waves.jpg";
import animalsBackground from "@/assets/animals-background.jpg";
import {
  audioApi,
  authApi,
  taxonomyApi,
  type AudioAnalysisResult,
  type AudioStream,
  type HumanToAnimalResult,
  type InterimAudioAnalysis,
  type SupportedAnimalsResponse,
  type TaxonomyResponse,
} from "@/services/api";

const RECORDER_MIME_TYPES = [
  "audio/webm;codecs=opus",
  "audio/webm",
  "audio/ogg;codecs=opus",
];

type PredictionSource = "zoovox_classifier" | "yamnet" | "heuristic" | undefined;

/**
 * Confidence is the model's own certainty in its prediction — never proof
 * the interpretation is objectively correct. A value from a real model
 * ("zoovox_classifier"/"yamnet") is labeled distinctly from a rule-based
 * estimate ("heuristic", or unknown source — treated conservatively as an
 * estimate rather than assumed to be a calibrated model score).
 */
function describeConfidence(value: number | null | undefined, source: PredictionSource) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return { badge: "N/A", toneClass: "bg-muted text-muted-foreground" };
  }
  const pct = Math.round(value * 100);
  const isRealModel = source === "zoovox_classifier" || source === "yamnet";
  const badge = `${isRealModel ? "Model" : "Est."} ${pct}%`;
  const toneClass =
    value >= 0.7
      ? "bg-primary/10 text-primary"
      : value >= 0.5
        ? "bg-amber-500/10 text-amber-600 dark:text-amber-400"
        : "bg-destructive/10 text-destructive";
  return { badge, toneClass };
}

type TaxonomyStatus = "idle" | "loading" | "found" | "not_found" | "unavailable";

const Dashboard = () => {
  const navigate = useNavigate();
  const [isRecordingAnimal, setIsRecordingAnimal] = useState(false);
  const [isFinalizingAnimal, setIsFinalizingAnimal] = useState(false);
  const [showMenu, setShowMenu] = useState(false);
  const [supportedAnimals, setSupportedAnimals] = useState<SupportedAnimalsResponse["animals"]>([]);
  const [analysisResult, setAnalysisResult] = useState<AudioAnalysisResult | null>(null);
  const [interimResult, setInterimResult] = useState<InterimAudioAnalysis | null>(null);
  const [selectedAnimal, setSelectedAnimal] = useState("dog");
  const [humanText, setHumanText] = useState("");
  const [humanToAnimalResult, setHumanToAnimalResult] = useState<HumanToAnimalResult | null>(null);
  const [isTranslating, setIsTranslating] = useState(false);
  const [taxonomyResult, setTaxonomyResult] = useState<TaxonomyResponse | null>(null);
  const [taxonomyStatus, setTaxonomyStatus] = useState<TaxonomyStatus>("idle");

  const recorderRef = useRef<MediaRecorder | null>(null);
  const microphoneRef = useRef<MediaStream | null>(null);
  const audioStreamRef = useRef<AudioStream | null>(null);
  const mountedRef = useRef(true);

  useEffect(() => {
    let cancelled = false;
    audioApi
      .getSupportedAnimals()
      .then((data) => {
        if (!cancelled) setSupportedAnimals(data.animals);
      })
      .catch((error) => {
        console.error("Unable to load supported animals", error);
        if (!cancelled) toast.error("Unable to load the supported-animal list.");
      });

    return () => {
      cancelled = true;
    };
  }, []);

  // Species/taxonomy reference lookup — fetched lazily only after a
  // translation result renders. This is a separate, unrelated GBIF lookup:
  // it never validates or contributes to the animal-sound classification
  // confidence shown above it.
  useEffect(() => {
    if (!analysisResult) {
      setTaxonomyResult(null);
      setTaxonomyStatus("idle");
      return;
    }

    let cancelled = false;
    setTaxonomyStatus("loading");
    setTaxonomyResult(null);

    taxonomyApi
      .lookupSpecies(analysisResult.animal_type)
      .then((data) => {
        if (cancelled) return;
        setTaxonomyResult(data);
        setTaxonomyStatus(data.found ? "found" : "not_found");
      })
      .catch((error) => {
        console.error("Species/taxonomy lookup failed", error);
        if (cancelled) return;
        setTaxonomyResult(null);
        setTaxonomyStatus("unavailable");
      });

    return () => {
      cancelled = true;
    };
  }, [analysisResult]);

  useEffect(() => {
    return () => {
      mountedRef.current = false;
      const recorder = recorderRef.current;
      if (recorder && recorder.state !== "inactive") {
        recorder.onstop = null;
        recorder.stop();
      }
      microphoneRef.current?.getTracks().forEach((track) => track.stop());
      audioStreamRef.current?.close();
    };
  }, []);

  const stopMicrophone = () => {
    microphoneRef.current?.getTracks().forEach((track) => track.stop());
    microphoneRef.current = null;
  };

  const stopAnimalRecording = () => {
    const recorder = recorderRef.current;
    if (!recorder || recorder.state === "inactive") return;

    setIsRecordingAnimal(false);
    setIsFinalizingAnimal(true);
    recorder.stop();
  };

  const startAnimalRecording = async () => {
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
      toast.error("This browser does not support live microphone recording.");
      return;
    }

    try {
      const microphone = await navigator.mediaDevices.getUserMedia({ audio: true });
      microphoneRef.current = microphone;
      const mimeType = RECORDER_MIME_TYPES.find((type) => MediaRecorder.isTypeSupported(type));
      const recorder = mimeType ? new MediaRecorder(microphone, { mimeType }) : new MediaRecorder(microphone);

      const liveStream = audioApi.createAudioStream(
        {
          onInterim: (result) => {
            if (mountedRef.current) setInterimResult(result);
          },
          onFinal: (result) => {
            if (!mountedRef.current) return;
            setAnalysisResult(result);
            setInterimResult(null);
            setIsRecordingAnimal(false);
            setIsFinalizingAnimal(false);
            audioStreamRef.current = null;
            toast.success(`Translation: “${result.translation_en}”`);
          },
          onError: (message) => {
            if (!mountedRef.current) return;
            setIsRecordingAnimal(false);
            setIsFinalizingAnimal(false);
            setInterimResult(null);
            toast.error(message);
            const activeRecorder = recorderRef.current;
            if (activeRecorder?.state === "recording") activeRecorder.stop();
          },
        },
        { contentType: recorder.mimeType || mimeType || "audio/webm" }
      );
      audioStreamRef.current = liveStream;

      await liveStream.ready;
      if (!mountedRef.current) {
        liveStream.close();
        stopMicrophone();
        return;
      }

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) liveStream.sendChunk(event.data);
      };
      recorder.onstop = () => {
        stopMicrophone();
        recorderRef.current = null;
        liveStream.end();
      };

      recorderRef.current = recorder;
      recorder.start(750);
      setAnalysisResult(null);
      setInterimResult(null);
      setIsRecordingAnimal(true);
      toast.success("Listening live — interim predictions will appear as you record.");
    } catch (error) {
      stopMicrophone();
      audioStreamRef.current?.close();
      audioStreamRef.current = null;
      if (mountedRef.current) {
        setIsRecordingAnimal(false);
        setIsFinalizingAnimal(false);
        toast.error(error instanceof Error ? error.message : "Unable to start live audio recording.");
      }
      console.error("Unable to start audio stream", error);
    }
  };

  const handleAnimalVoice = async () => {
    if (isRecordingAnimal) {
      stopAnimalRecording();
      return;
    }
    await startAnimalRecording();
  };

  const handleHumanTranslation = async () => {
    const text = humanText.trim();
    if (!text) return;

    setIsTranslating(true);
    try {
      const result = await audioApi.humanToAnimal(text, selectedAnimal);
      setHumanToAnimalResult(result);
      toast.success(`Created a ${selectedAnimal} communication cue.`);
      if (result.audio_url) {
        void new Audio(result.audio_url).play().catch(() => {
          toast.info("The cue is ready, but your browser could not play its audio URL.");
        });
      }
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Translation failed. Please try again.");
      console.error("Human-to-animal translation failed", error);
    } finally {
      setIsTranslating(false);
    }
  };

  const handleLogout = () => {
    recorderRef.current?.stop();
    audioStreamRef.current?.close();
    authApi.logout();
    navigate("/", { replace: true });
  };

  return (
    <div className="min-h-screen relative overflow-hidden">
      <div
        className="absolute inset-0 bg-cover bg-center opacity-10 animate-float"
        style={{ backgroundImage: `url(${animalsBackground})` }}
      />
      <div className="absolute inset-0 bg-background" />
      <div className="absolute top-40 right-40 w-96 h-96 bg-primary/10 rounded-full blur-3xl animate-glow" />
      <div
        className="absolute bottom-20 left-20 w-80 h-80 bg-accent/10 rounded-full blur-3xl animate-glow"
        style={{ animationDelay: "2s" }}
      />

      <div className="relative z-10 min-h-screen flex flex-col">
        <header className="p-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <PawPrint className="w-7 h-7 text-primary" />
            <span className="text-xl font-bold">ZOOVOX</span>
          </div>
          <div className="flex items-center gap-2">
            {showMenu && (
              <div className="absolute right-4 top-12 bg-card border rounded-lg shadow-lg p-2 w-52 animate-slide-down">
                <button
                  onClick={() => navigate("/veterinary")}
                  className="w-full flex items-center gap-2 px-3 py-2 text-sm hover:bg-accent rounded"
                >
                  <MapPin className="w-4 h-4" />
                  Veterinary care
                </button>
                <button
                  onClick={() => navigate("/history")}
                  className="w-full flex items-center gap-2 px-3 py-2 text-sm hover:bg-accent rounded"
                >
                  <Clock className="w-4 h-4" />
                  Translation history
                </button>
                <button
                  onClick={handleLogout}
                  className="w-full flex items-center gap-2 px-3 py-2 text-sm hover:bg-accent rounded"
                >
                  <LogOut className="w-4 h-4" />
                  Log out
                </button>
              </div>
            )}
            <Button variant="ghost" size="icon" onClick={() => setShowMenu((open) => !open)} aria-label="Open menu">
              <Menu className="w-5 h-5" />
            </Button>
          </div>
        </header>

        <main className="flex-1 flex flex-col items-center justify-center px-4 py-6">
          <div className="relative w-full max-w-md mb-8">
            <div className="aspect-square rounded-2xl overflow-hidden bg-muted/50 border">
              <img
                src={voiceWaves}
                alt="Voice waves visualization"
                className="w-full h-full object-cover"
                style={{
                  opacity: isRecordingAnimal ? 1 : 0.4,
                  filter: isRecordingAnimal ? "none" : "grayscale(100%)",
                }}
              />
              {isRecordingAnimal && (
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="flex gap-1 items-end h-20">
                    {[...Array(20)].map((_, index) => (
                      <div
                        key={index}
                        className="w-1 bg-primary rounded animate-equalizer"
                        style={{
                          height: `${20 + ((index * 37) % 80)}%`,
                          animationDelay: `${index * 50}ms`,
                          animationDuration: `${300 + ((index * 71) % 400)}ms`,
                        }}
                      />
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>

          <Button
            onClick={handleAnimalVoice}
            disabled={isFinalizingAnimal}
            size="lg"
            className={`w-full max-w-md gap-3 transition-all ${
              isRecordingAnimal ? "bg-primary ring-2 ring-primary ring-offset-2" : ""
            }`}
          >
            <Mic className={`w-6 h-6 ${isRecordingAnimal ? "animate-pulse" : ""}`} />
            <span className="text-lg">
              {isRecordingAnimal
                ? "Stop listening"
                : isFinalizingAnimal
                  ? "Analyzing live recording…"
                  : "Listen to animal"}
            </span>
          </Button>

          {interimResult && (
            <Card className="w-full max-w-md mt-4 p-3 text-sm border-primary/40">
              <p className="text-muted-foreground">Live prediction</p>
              <p className="font-medium capitalize flex flex-wrap items-center gap-1.5">
                {interimResult.animal_type} · {interimResult.emotion}
                <span
                  className={`text-xs px-1.5 rounded normal-case ${describeConfidence(interimResult.confidence, undefined).toneClass}`}
                  title="Model confidence reflects the model's certainty in its own prediction, not a guarantee it's correct."
                >
                  {describeConfidence(interimResult.confidence, undefined).badge}
                </span>
              </p>
            </Card>
          )}

          {analysisResult && (
            <Card className="w-full max-w-md mt-6 p-4 animate-fade-in">
              <div className="flex items-center justify-between mb-2">
                <h3 className="font-semibold">Translation result</h3>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => setAnalysisResult(null)}
                  className="text-muted-foreground"
                  aria-label="Dismiss translation result"
                >
                  <X className="w-4 h-4" />
                </Button>
              </div>
              <div className="space-y-2 text-sm">
                <p className="font-medium">“{analysisResult.translation_en}”</p>
                <div className="flex flex-wrap items-center gap-4 text-muted-foreground">
                  <span className="flex items-center gap-1 capitalize">
                    <PawPrint className="w-3 h-3" />
                    {analysisResult.animal_type}
                    <span
                      className={`ml-1 text-xs px-1.5 rounded normal-case ${describeConfidence(analysisResult.animal_confidence, analysisResult.prediction_source).toneClass}`}
                      title="Model confidence reflects the model's certainty in its own prediction, not a guarantee it's correct."
                    >
                      {describeConfidence(analysisResult.animal_confidence, analysisResult.prediction_source).badge}
                    </span>
                  </span>
                  <span className="flex items-center gap-1 capitalize">
                    <Volume2 className="w-3 h-3" />
                    {analysisResult.detected_emotion}
                    <span
                      className={`ml-1 text-xs px-1.5 rounded normal-case ${describeConfidence(analysisResult.emotion_confidence, "heuristic").toneClass}`}
                      title="Emotion confidence is a rule-based estimate, not output from a trained model — treat it as a rough signal, not a guarantee."
                    >
                      {describeConfidence(analysisResult.emotion_confidence, "heuristic").badge}
                    </span>
                  </span>
                </div>
                <p className="text-xs text-muted-foreground">{analysisResult.behavioral_context}</p>
                <p className="text-xs text-muted-foreground/80 italic">
                  Confidence reflects the model's certainty in its own prediction — not a guarantee the interpretation is correct.
                </p>

                <div className="pt-2 mt-1 border-t border-border/40 text-xs text-muted-foreground/80">
                  {taxonomyStatus === "loading" && <p>Looking up species reference…</p>}

                  {taxonomyStatus === "found" && taxonomyResult && (
                    <div className="space-y-0.5">
                      <p className="flex items-center gap-1 flex-wrap">
                        <BookOpen className="w-3 h-3 shrink-0" />
                        <span>
                          Species reference:{" "}
                          <em className="not-italic font-medium">
                            {taxonomyResult.canonical_name ?? taxonomyResult.scientific_name}
                          </em>
                          {taxonomyResult.rank && <span className="capitalize"> ({taxonomyResult.rank})</span>}
                        </span>
                      </p>
                      {taxonomyResult.match_type === "HIGHERRANK" && (
                        <p className="pl-4">Higher-rank match only — not identified to species level.</p>
                      )}
                      <p className="pl-4">
                        Source:{" "}
                        {taxonomyResult.source_url ? (
                          <a
                            href={taxonomyResult.source_url}
                            target="_blank"
                            rel="noreferrer"
                            className="underline underline-offset-2 hover:text-foreground"
                          >
                            {taxonomyResult.attribution ?? "GBIF"}
                          </a>
                        ) : (
                          taxonomyResult.attribution ?? "GBIF"
                        )}
                      </p>
                    </div>
                  )}

                  {taxonomyStatus === "not_found" && (
                    <p>No species/taxonomy reference found for “{analysisResult.animal_type}”.</p>
                  )}

                  {taxonomyStatus === "unavailable" && <p>Species reference lookup is temporarily unavailable.</p>}
                </div>
              </div>
            </Card>
          )}

          <div className="w-full max-w-md mt-8 space-y-4">
            <h3 className="text-center font-semibold text-muted-foreground">Human → Animal</h3>
            <Card className="p-4">
              <div className="space-y-3">
                <div>
                  <label htmlFor="human-text" className="text-sm font-medium mb-1 block">
                    What to say
                  </label>
                  <input
                    id="human-text"
                    type="text"
                    value={humanText}
                    onChange={(event) => setHumanText(event.target.value)}
                    placeholder="e.g., Come here, good boy!"
                    maxLength={500}
                    className="w-full p-2 border rounded bg-background"
                  />
                </div>
                <div>
                  <label htmlFor="target-animal" className="text-sm font-medium mb-1 block">
                    Target animal
                  </label>
                  <select
                    id="target-animal"
                    value={selectedAnimal}
                    onChange={(event) => setSelectedAnimal(event.target.value)}
                    className="w-full p-2 border rounded bg-background"
                  >
                    {supportedAnimals.length ? (
                      supportedAnimals.map((animal) => (
                        <option key={animal.id} value={animal.id}>
                          {animal.emoji} {animal.name}
                        </option>
                      ))
                    ) : (
                      <option value="dog">🐕 Dog</option>
                    )}
                  </select>
                </div>
                <Button onClick={handleHumanTranslation} disabled={isTranslating || !humanText.trim()} className="w-full">
                  {isTranslating ? "Creating cue…" : "Translate & play"}
                </Button>
              </div>
            </Card>

            {humanToAnimalResult && (
              <Card className="p-4 animate-fade-in border-primary">
                <h4 className="font-semibold mb-2">Animal communication cue</h4>
                <p className="text-sm mb-2">{humanToAnimalResult.animal_cue_description}</p>
                <div className="flex flex-wrap gap-1">
                  {humanToAnimalResult.recommended_actions.map((action, index) => (
                    <span key={index} className="text-xs bg-primary/10 text-primary px-2 py-1 rounded-full">
                      {action}
                    </span>
                  ))}
                </div>
              </Card>
            )}
          </div>
        </main>
      </div>
    </div>
  );
};

export default Dashboard;
