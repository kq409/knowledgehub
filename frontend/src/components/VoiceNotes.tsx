import { useCallback, useEffect, useRef, useState } from 'react';
import styles from './VoiceNotes.module.css';
import { RecordButton } from './RecordButton';
import { UploadZone } from './UploadZone';
import { TextInputZone } from './TextInputZone';
import { SettingsPanel } from './SettingsPanel';
import { TranscriptionResults } from './TranscriptionResults';
import { ErrorMessage } from './ErrorMessage';
import { VoiceNoteCard } from './VoiceNoteCard';
import { VoiceNoteList } from './VoiceNoteList';
import type { ReviewStatus, VoiceNote, VoiceNoteDraft } from '../types';
import { useAppStatus } from '../hooks/appStatus';

interface TranscriptionResponse {
  success: boolean;
  text?: string;
  error?: string;
}

interface SystemPromptResponse {
  default_prompt: string;
}

const LLM_CLEANING_ERROR =
  'LLM cleaning failed. Your transcription is unaffected — check the backend terminal for the detailed error.';

const NOTE_EXTRACT_ERROR =
  'Could not extract a structured research note. Your transcription is unaffected.';

export function VoiceNotes() {
  const { whisper_enabled: whisperEnabled, uploads_enabled: uploadsEnabled } =
    useAppStatus();
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [rawText, setRawText] = useState<string | null>(null);
  const [cleanedText, setCleanedText] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [useLLM, setUseLLM] = useState(true);
  const [isCopied, setIsCopied] = useState(false);
  const [systemPrompt, setSystemPrompt] = useState('');
  const [isLoadingPrompt, setIsLoadingPrompt] = useState(true);
  const [isDragging, setIsDragging] = useState(false);
  const [isCleaningWithLLM, setIsCleaningWithLLM] = useState(false);
  const [isOriginalExpanded, setIsOriginalExpanded] = useState(true);
  const [currentNote, setCurrentNote] = useState<VoiceNoteDraft | null>(null);
  const [savedNotes, setSavedNotes] = useState<VoiceNote[]>([]);
  const [isExtracting, setIsExtracting] = useState(false);
  const [isSaving, setIsSaving] = useState(false);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const isKeyDownRef = useRef(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const loadSavedNotes = useCallback(async () => {
    try {
      const response = await fetch('/api/notes?source_type=voice');
      if (!response.ok) {
        return;
      }
      const notes = (await response.json()) as VoiceNote[];
      setSavedNotes(notes);
    } catch (err) {
      console.error('Failed to load voice notes:', err);
    }
  }, []);

  useEffect(() => {
    const loadSystemPrompt = async () => {
      try {
        const response = await fetch('/api/system-prompt');
        const data = (await response.json()) as SystemPromptResponse;
        setSystemPrompt(data.default_prompt);
      } catch (err) {
        console.error('Failed to load system prompt:', err);
        setError('Failed to load system prompt');
      } finally {
        setIsLoadingPrompt(false);
      }
    };

    void loadSystemPrompt();
    void loadSavedNotes();
  }, [loadSavedNotes]);

  const extractStructuredNote = useCallback(
    async (raw: string, cleaned: string) => {
      setIsExtracting(true);
      setCurrentNote(null);

      try {
        const response = await fetch('/api/voice-notes/extract', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            raw_text: raw,
            cleaned_text: cleaned,
          }),
        });

        if (!response.ok) {
          setError(NOTE_EXTRACT_ERROR);
          return;
        }

        const extracted = (await response.json()) as VoiceNoteDraft;
        setCurrentNote({
          ...extracted,
          raw_transcript: raw,
          cleaned_transcript: cleaned,
          review_status: 'generated',
        });
      } catch {
        setError(NOTE_EXTRACT_ERROR);
      } finally {
        setIsExtracting(false);
      }
    },
    []
  );

  const streamCleanText = useCallback(
    async (text: string) => {
      setIsCleaningWithLLM(true);
      setCleanedText('');
      setCurrentNote(null);

      try {
        const cleanResponse = await fetch('/api/clean', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            text,
            ...(systemPrompt && { system_prompt: systemPrompt }),
          }),
        });

        if (!cleanResponse.ok || !cleanResponse.body) {
          setIsCleaningWithLLM(false);
          setError(LLM_CLEANING_ERROR);
          return;
        }

        const reader = cleanResponse.body.getReader();
        const decoder = new TextDecoder();
        let accumulated = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) {
            break;
          }
          accumulated += decoder.decode(value, { stream: true });
          setCleanedText(accumulated);
        }

        accumulated += decoder.decode();
        const cleaned = accumulated.trim();
        setCleanedText(cleaned);
        setIsCleaningWithLLM(false);

        if (cleaned) {
          await extractStructuredNote(text, cleaned);
        }
      } catch {
        setIsCleaningWithLLM(false);
        setError(LLM_CLEANING_ERROR);
      }
    },
    [systemPrompt, extractStructuredNote]
  );

  const uploadAudio = useCallback(
    async (audioBlob: Blob) => {
      const formData = new FormData();
      formData.append('audio', audioBlob, 'recording.webm');

      try {
        const transcribeResponse = await fetch('/api/transcribe', {
          method: 'POST',
          body: formData,
        });

        if (!transcribeResponse.ok) {
          throw new Error(
            `Transcription failed: ${transcribeResponse.statusText}`
          );
        }

        const transcribeData =
          (await transcribeResponse.json()) as TranscriptionResponse;

        if (!transcribeData.success) {
          throw new Error(transcribeData.error || 'Transcription failed');
        }

        setRawText(transcribeData.text || '');
        setIsProcessing(false);
        setError(null);

        if (useLLM && transcribeData.text) {
          await streamCleanText(transcribeData.text);
        }
      } catch (err) {
        const errorMessage =
          err instanceof Error ? err.message : 'Unknown error';
        setError('Processing failed: ' + errorMessage);
        setIsProcessing(false);
      }
    },
    [useLLM, streamCleanText]
  );

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      mediaRecorderRef.current = new MediaRecorder(stream);
      chunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (e: BlobEvent) => {
        chunksRef.current.push(e.data);
      };

      mediaRecorderRef.current.onstop = async () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
        await uploadAudio(blob);
        stream.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
      setError(null);
      setRawText(null);
      setCleanedText(null);
      setCurrentNote(null);
      setIsCleaningWithLLM(false);
      setIsExtracting(false);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      setError('Microphone access denied: ' + errorMessage);
    }
  }, [uploadAudio]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      setIsProcessing(true);
    }
  }, [isRecording]);

  useEffect(() => {
    return () => {
      if (mediaRecorderRef.current?.state === 'recording') {
        mediaRecorderRef.current.stop();
      }
      streamRef.current?.getTracks().forEach((track) => track.stop());
    };
  }, []);

  const processAudioFile = (file: File) => {
    if (!file) return;

    if (!file.type.startsWith('audio/')) {
      setError('Please select an audio file');
      return;
    }

    setError(null);
    setRawText(null);
    setCleanedText(null);
    setCurrentNote(null);
    setIsProcessing(true);
    setIsCleaningWithLLM(false);
    setIsExtracting(false);

    const blob = new Blob([file], { type: file.type });
    void uploadAudio(blob);
  };

  const handleDragEnter = () => {
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (file: File) => {
    if (isProcessing || isRecording) return;
    processAudioFile(file);
  };

  const handleFileSelect = (file: File) => {
    processAudioFile(file);

    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleTextSubmit = useCallback(
    async (text: string) => {
      if (!text.trim()) return;

      try {
        setError(null);
        setRawText(null);
        setCleanedText(null);
        setCurrentNote(null);
        setIsProcessing(true);
        setIsCleaningWithLLM(false);
        setIsExtracting(false);

        setRawText(text);
        setIsProcessing(false);

        if (useLLM) {
          await streamCleanText(text);
        }
      } catch (err) {
        const errorMessage =
          err instanceof Error ? err.message : 'Unknown error';
        setError('Processing failed: ' + errorMessage);
        setIsProcessing(false);
        setIsCleaningWithLLM(false);
      }
    },
    [useLLM, streamCleanText]
  );

  const copyToClipboard = (text: string) => {
    navigator.clipboard
      .writeText(text)
      .then(() => {
        setIsCopied(true);
        setTimeout(() => setIsCopied(false), 2000);
      })
      .catch((err: Error) => setError('Copy failed: ' + err.message));
  };

  const saveCurrentNote = useCallback(async () => {
    if (!currentNote) return;

    setIsSaving(true);
    try {
      const isUpdate = Boolean(currentNote.id);
      const url = isUpdate ? `/api/notes/${currentNote.id}` : '/api/notes';
      const response = await fetch(url, {
        method: isUpdate ? 'PATCH' : 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(
          isUpdate
            ? {
                title: currentNote.title,
                summary: currentNote.summary,
                observations: currentNote.observations,
                hypotheses: currentNote.hypotheses,
                questions: currentNote.questions,
                next_steps: currentNote.next_steps,
                tags: currentNote.tags,
                review_status: currentNote.review_status,
                paper_ids: currentNote.paper_ids ?? [],
              }
            : {
                ...currentNote,
                paper_ids: currentNote.paper_ids ?? [],
              }
        ),
      });

      if (!response.ok) {
        throw new Error('Failed to save voice note');
      }

      const saved = (await response.json()) as VoiceNote;
      setCurrentNote(saved);
      await loadSavedNotes();
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      setError('Save failed: ' + errorMessage);
    } finally {
      setIsSaving(false);
    }
  }, [currentNote, loadSavedNotes]);

  const setNoteStatus = useCallback(
    async (status: ReviewStatus) => {
      if (!currentNote) return;

      const nextNote = { ...currentNote, review_status: status };
      setCurrentNote(nextNote);

      if (!currentNote.id) {
        return;
      }

      try {
        const response = await fetch(`/api/notes/${currentNote.id}`, {
          method: 'PATCH',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ review_status: status }),
        });
        if (!response.ok) {
          throw new Error('Failed to update review status');
        }
        const saved = (await response.json()) as VoiceNote;
        setCurrentNote(saved);
        await loadSavedNotes();
      } catch (err) {
        const errorMessage =
          err instanceof Error ? err.message : 'Unknown error';
        setError('Status update failed: ' + errorMessage);
      }
    },
    [currentNote, loadSavedNotes]
  );

  const deleteNote = useCallback(
    async (id: string) => {
      try {
        const response = await fetch(`/api/notes/${id}`, {
          method: 'DELETE',
        });
        if (!response.ok) {
          throw new Error('Failed to delete voice note');
        }
        if (currentNote?.id === id) {
          setCurrentNote(null);
        }
        await loadSavedNotes();
      } catch (err) {
        const errorMessage =
          err instanceof Error ? err.message : 'Unknown error';
        setError('Delete failed: ' + errorMessage);
      }
    },
    [currentNote, loadSavedNotes]
  );

  useEffect(() => {
    if (!whisperEnabled) {
      return;
    }
    const handleKeyDown = (e: KeyboardEvent) => {
      if (isProcessing || e.repeat || isKeyDownRef.current) return;

      const target = e.target as HTMLElement;
      if (
        e.key.toLowerCase() === 'v' &&
        !['INPUT', 'TEXTAREA'].includes(target.tagName)
      ) {
        e.preventDefault();
        isKeyDownRef.current = true;

        if (!isRecording) {
          void startRecording();
        }
      }
    };

    const handleKeyUp = (e: KeyboardEvent) => {
      if (e.key.toLowerCase() === 'v') {
        isKeyDownRef.current = false;

        if (isRecording) {
          stopRecording();
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, [
    whisperEnabled,
    isRecording,
    isProcessing,
    startRecording,
    stopRecording,
  ]);

  return (
    <div className={styles.voiceNotes}>
      {whisperEnabled ? (
        <RecordButton
          isRecording={isRecording}
          isProcessing={isProcessing}
          onStartRecording={startRecording}
          onStopRecording={stopRecording}
        />
      ) : (
        <p className={styles.hint}>
          Microphone recording is disabled on this deployment.
        </p>
      )}

      {whisperEnabled && uploadsEnabled ? (
        <UploadZone
          isProcessing={isProcessing}
          isDragging={isDragging}
          onFileSelect={handleFileSelect}
          onDragEnter={handleDragEnter}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          fileInputRef={fileInputRef}
        />
      ) : null}

      <TextInputZone
        isProcessing={isProcessing}
        onTextSubmit={handleTextSubmit}
      />

      <SettingsPanel
        useLLM={useLLM}
        systemPrompt={systemPrompt}
        isLoadingPrompt={isLoadingPrompt}
        onToggleLLM={setUseLLM}
        onPromptChange={setSystemPrompt}
      />

      {error && (
        <ErrorMessage message={error} onDismiss={() => setError(null)} />
      )}

      <TranscriptionResults
        rawText={rawText}
        cleanedText={cleanedText}
        useLLM={useLLM}
        isCopied={isCopied}
        isCleaningWithLLM={isCleaningWithLLM}
        isProcessing={isProcessing}
        isOriginalExpanded={isOriginalExpanded}
        onCopy={copyToClipboard}
        onToggleOriginalExpanded={() =>
          setIsOriginalExpanded(!isOriginalExpanded)
        }
      />

      {(isExtracting || currentNote) && (
        <VoiceNoteCard
          note={
            currentNote ?? {
              title: '',
              summary: '',
              observations: [],
              hypotheses: [],
              questions: [],
              next_steps: [],
              tags: [],
              raw_transcript: rawText || '',
              cleaned_transcript: cleanedText || '',
              review_status: 'generated',
            }
          }
          isSaving={isSaving}
          isExtracting={isExtracting}
          isCopied={isCopied}
          onChange={(note) => setCurrentNote(note)}
          onSave={() => void saveCurrentNote()}
          onCopy={copyToClipboard}
          onSetStatus={(status) => void setNoteStatus(status)}
        />
      )}

      <VoiceNoteList
        notes={savedNotes}
        selectedId={currentNote?.id}
        onSelect={(note) => {
          setCurrentNote(note);
          setRawText(note.raw_transcript);
          setCleanedText(note.cleaned_transcript);
          setError(null);
        }}
        onDelete={(id) => void deleteNote(id)}
      />
    </div>
  );
}
