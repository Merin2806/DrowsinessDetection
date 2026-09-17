package com.driveguard.alert;

import javax.sound.sampled.*;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Controls a single-instance audible warning alert.
 *
 * Design decisions:
 * ─ The Clip is created once at startup (not per frame) to avoid overhead.
 * ─ startAlert() / stopAlert() are guarded by AtomicBoolean so they are
 *   safe to call on every video frame without creating duplicate sounds.
 * ─ Audio is generated programmatically (880 Hz sine wave) — no audio file
 *   dependency.
 * ─ The Clip's internal playback thread is managed by the Java sound system;
 *   this class does NOT create any threads and will NOT freeze the UI.
 */
public class AlertService {

    private Clip alertClip;
    private final AtomicBoolean alertPlaying = new AtomicBoolean(false);

    public AlertService() {
        try {
            alertClip = buildBeepClip(880.0, 600);  // 880 Hz, 600 ms segment
        } catch (Exception e) {
            // Non-fatal — visual alert still works without audio
            System.err.println("[AlertService] Audio init failed (continuing without sound): "
                    + e.getMessage());
        }
    }

    /**
     * Start the looping alert.
     * Idempotent — safe to call every frame while DROWSY.
     */
    public void startAlert() {
        if (alertClip != null && alertPlaying.compareAndSet(false, true)) {
            alertClip.setFramePosition(0);
            alertClip.loop(Clip.LOOP_CONTINUOUSLY);
        }
    }

    /**
     * Stop the looping alert.
     * Idempotent — safe to call every frame while ACTIVE.
     */
    public void stopAlert() {
        if (alertClip != null && alertPlaying.compareAndSet(true, false)) {
            alertClip.stop();
        }
    }

    /** Release audio resources — call on application exit. */
    public void shutdown() {
        stopAlert();
        if (alertClip != null) {
            alertClip.close();
        }
    }

    public boolean isPlaying() {
        return alertPlaying.get();
    }

    // ── Audio generation ─────────────────────────────────────────────────────

    /**
     * Synthesise a sine-wave warning beep as a loopable PCM Clip.
     *
     * @param frequencyHz  pitch of the beep (880 Hz = high A — audible warning tone)
     * @param durationMs   length of one loop segment in milliseconds
     */
    private Clip buildBeepClip(double frequencyHz, int durationMs) throws LineUnavailableException {
        float sampleRate   = 44100f;
        int   numSamples   = (int) (sampleRate * durationMs / 1000.0);
        byte[] pcmData     = new byte[numSamples * 2]; // 16-bit mono = 2 bytes/sample

        for (int i = 0; i < numSamples; i++) {
            double angle = 2.0 * Math.PI * i * frequencyHz / sampleRate;

            // Apply a short fade-in and fade-out envelope to prevent audible clicks
            // when the loop restarts.
            double fadeSamples = numSamples * 0.04; // 4% fade on each end
            double envelope = 1.0;
            if (i < fadeSamples) {
                envelope = i / fadeSamples;
            } else if (i > numSamples - fadeSamples) {
                envelope = (numSamples - i) / fadeSamples;
            }

            short sample = (short) (Short.MAX_VALUE * 0.65 * envelope * Math.sin(angle));

            // Little-endian 16-bit PCM (required by Java Sound default format)
            pcmData[i * 2]     = (byte) (sample & 0xFF);
            pcmData[i * 2 + 1] = (byte) ((sample >> 8) & 0xFF);
        }

        AudioFormat format = new AudioFormat(
                sampleRate,  // sample rate
                16,          // bits per sample
                1,           // channels (mono)
                true,        // signed
                false        // little-endian
        );

        DataLine.Info info = new DataLine.Info(Clip.class, format);
        Clip clip = (Clip) AudioSystem.getLine(info);
        clip.open(format, pcmData, 0, pcmData.length);
        return clip;
    }
}
