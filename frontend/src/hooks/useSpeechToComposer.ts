import { useCallback, useEffect, useRef, useState } from 'react';

interface SpeechRecognitionLike {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  onresult: ((event: SpeechRecognitionEventLike) => void) | null;
  onerror: ((event: { error: string }) => void) | null;
  onend: (() => void) | null;
  start: () => void;
  stop: () => void;
}

interface SpeechRecognitionEventLike {
  results: ArrayLike<{
    isFinal: boolean;
    0: { transcript: string };
  }>;
}

type SpeechRecognitionCtor = new () => SpeechRecognitionLike;

function getSpeechRecognition(): SpeechRecognitionCtor | null {
  const speechWindow = window as Window & {
    SpeechRecognition?: SpeechRecognitionCtor;
    webkitSpeechRecognition?: SpeechRecognitionCtor;
  };
  return (
    speechWindow.SpeechRecognition ??
    speechWindow.webkitSpeechRecognition ??
    null
  );
}

function joinTranscript(...parts: string[]): string {
  return parts
    .map((part) => part.trim())
    .filter(Boolean)
    .join(' ');
}

export function useSpeechToComposer(
  value: string,
  onChange: (next: string) => void
) {
  const [isListening, setIsListening] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const valueRef = useRef(value);
  valueRef.current = value;
  const prefixRef = useRef('');
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);

  const supportsLive =
    typeof window !== 'undefined' && getSpeechRecognition() != null;

  const stopLive = useCallback(() => {
    recognitionRef.current?.stop();
    recognitionRef.current = null;
    setIsListening(false);
  }, []);

  const startLive = useCallback(() => {
    const Ctor = getSpeechRecognition();
    if (!Ctor) {
      return;
    }
    prefixRef.current = valueRef.current.trim();
    const recognition = new Ctor();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = navigator.language || 'en-US';
    recognition.onresult = (event) => {
      let sessionFinal = '';
      let interim = '';
      for (let i = 0; i < event.results.length; i += 1) {
        const result = event.results[i];
        const text = result?.[0]?.transcript ?? '';
        if (result?.isFinal) {
          sessionFinal += text;
        } else {
          interim += text;
        }
      }
      onChange(joinTranscript(prefixRef.current, sessionFinal, interim));
    };
    recognition.onerror = (event) => {
      if (event.error !== 'aborted' && event.error !== 'no-speech') {
        setError(`Voice input failed: ${event.error}`);
      }
      setIsListening(false);
    };
    recognition.onend = () => {
      recognitionRef.current = null;
      setIsListening(false);
    };
    recognitionRef.current = recognition;
    recognition.start();
    setIsListening(true);
    setError(null);
  }, [onChange]);

  const stopWhisper = useCallback(() => {
    if (mediaRecorderRef.current?.state === 'recording') {
      mediaRecorderRef.current.stop();
    }
    setIsListening(false);
  }, []);

  const startWhisper = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const recorder = new MediaRecorder(stream);
      chunksRef.current = [];
      recorder.ondataavailable = (event: BlobEvent) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };
      recorder.onstop = async () => {
        stream.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
        if (blob.size === 0) {
          setIsTranscribing(false);
          return;
        }
        setIsTranscribing(true);
        try {
          const formData = new FormData();
          formData.append('audio', blob, 'composer.webm');
          const response = await fetch('/api/transcribe', {
            method: 'POST',
            body: formData,
          });
          const data = (await response.json()) as {
            success?: boolean;
            text?: string;
            error?: string;
          };
          if (!response.ok || !data.success) {
            throw new Error(data.error || response.statusText);
          }
          onChange(joinTranscript(valueRef.current, data.text || ''));
          setError(null);
        } catch (err) {
          const message = err instanceof Error ? err.message : 'Unknown error';
          setError('Transcription failed: ' + message);
        } finally {
          setIsTranscribing(false);
        }
      };
      mediaRecorderRef.current = recorder;
      recorder.start();
      setIsListening(true);
      setError(null);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      setError('Microphone access denied: ' + message);
    }
  }, [onChange]);

  const stop = useCallback(() => {
    if (!isListening) {
      return;
    }
    if (supportsLive) {
      stopLive();
    } else {
      stopWhisper();
    }
  }, [isListening, supportsLive, stopLive, stopWhisper]);

  const toggle = useCallback(() => {
    if (isListening) {
      stop();
      return;
    }
    if (supportsLive) {
      startLive();
    } else {
      void startWhisper();
    }
  }, [isListening, supportsLive, startLive, startWhisper, stop]);

  useEffect(() => {
    return () => {
      recognitionRef.current?.stop();
      if (mediaRecorderRef.current?.state === 'recording') {
        mediaRecorderRef.current.stop();
      }
      streamRef.current?.getTracks().forEach((track) => track.stop());
    };
  }, []);

  return {
    isListening,
    isTranscribing,
    error,
    toggle,
    stop,
    supportsLive,
    dismissError: () => setError(null),
  };
}
