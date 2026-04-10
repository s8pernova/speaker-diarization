# Speaker Diarization

Batch tool that takes meeting recordings, runs speaker diarization with pyannoteAI, cuts the audio by speaker timestamps, and exports grouped speaker packs like `speaker_00.wav`, `speaker_01.wav`, and a review manifest.

This project does **not** try to identify real names. It only answers **who spoke when**, groups those chunks by generic speaker label, and leaves the human naming step for later.

## What this tool does

Given one or more files like:

- `.mp4`
- `.mp3`
- `.wav`
- `.m4a`

the tool will:

1. extract audio from each input file
2. upload that audio to pyannoteAI
3. submit a diarization job
4. fetch speaker segments like `SPEAKER_00`, `SPEAKER_01`
5. cut those time ranges from the source audio
6. group the clips by speaker
7. merge each speaker’s clips into a single file
8. export a manifest for review and renaming

## Planned workflow

```text
input media
  -> extract audio
  -> upload to pyannoteAI
  -> submit diarization
  -> poll job result
  -> choose diarization or exclusiveDiarization
  -> cut segments
  -> group by speaker
  -> merge per speaker
  -> export packs + manifest
```
