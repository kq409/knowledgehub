---
name: voice-cleanup
description: Conventions for cleaning raw voice transcripts (reference for chat; Voice Settings still drive /api/clean).
---

# Voice transcript cleanup

When advising on or discussing transcript cleanup:

- Remove filler words (um, uh, like, you know, basically, actually, etc.)
- Remove redundant statements and rambling
- Fix grammar and speech-to-text errors
- Preserve key points, technical details, names, numbers, and action items
- Use proper punctuation and maintain the speaker's tone
- Return only cleaned text with no preamble

The Voice tab Settings system prompt still controls `/api/clean` directly. This skill is guidance for the chat agent, not a replacement for that setting.
