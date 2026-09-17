package com.driveguard.detection;

import com.driveguard.model.DetectionResult;
import com.driveguard.model.DetectionResult.DriverStatus;
import com.driveguard.model.DetectionResult.EyeStatus;

/**
 * Stateful drowsiness detector.
 *
 * Maintains a consecutive-closed-eye frame counter.
 * Only transitions to DROWSY after the counter exceeds CLOSED_EYE_THRESHOLD.
 *
 * ── Threshold guidance ──────────────────────────────────────────────────────
 *   CLOSED_EYE_THRESHOLD = 20  →  ~0.7 s closure at 30 FPS  (default)
 *   CLOSED_EYE_THRESHOLD = 15  →  ~0.5 s (more sensitive)
 *   CLOSED_EYE_THRESHOLD = 30  →  ~1.0 s (less sensitive)
 *
 *   Adjust after testing on the demo machine. Higher = fewer false alerts.
 * ────────────────────────────────────────────────────────────────────────────
 */
public class DrowsinessDetector {

    // ── CONFIGURABLE THRESHOLD — change this one constant to tune sensitivity ──
    private static final int CLOSED_EYE_THRESHOLD = 20;
    // ─────────────────────────────────────────────────────────────────────────

    private int closedFrameCount = 0;
    private DriverStatus currentStatus = DriverStatus.WAITING;

    /**
     * Process the detection result for one video frame.
     *
     * @param faceDetected true if a face was found in this frame
     * @param eyesDetected true if at least one eye was found inside the face ROI
     * @return a DetectionResult capturing the updated state
     */
    public DetectionResult process(boolean faceDetected, boolean eyesDetected) {

        // ── No face ─────────────────────────────────────────────────────────
        if (!faceDetected) {
            // Reset everything; we cannot judge drowsiness without seeing the driver
            closedFrameCount = 0;
            currentStatus    = DriverStatus.WAITING;
            return new DetectionResult(false, EyeStatus.NOT_DETECTED, DriverStatus.WAITING, 0);
        }

        // ── Face present, eyes visible ───────────────────────────────────────
        if (eyesDetected) {
            // Reset the counter — driver is alert
            closedFrameCount = 0;
            currentStatus    = DriverStatus.ACTIVE;
            return new DetectionResult(true, EyeStatus.OPEN, DriverStatus.ACTIVE, 0);
        }

        // ── Face present, eyes NOT detected ─────────────────────────────────
        // Could be a real closure or a momentary missed detection. Accumulate.
        closedFrameCount++;

        if (closedFrameCount >= CLOSED_EYE_THRESHOLD) {
            // Threshold exceeded → drowsiness confirmed
            currentStatus = DriverStatus.DROWSY;
        } else {
            // Still within the tolerance window — treat as ACTIVE
            currentStatus = DriverStatus.ACTIVE;
        }

        return new DetectionResult(true, EyeStatus.CLOSED, currentStatus, closedFrameCount);
    }

    /** Reset detector state (called when camera restarts). */
    public void reset() {
        closedFrameCount = 0;
        currentStatus    = DriverStatus.WAITING;
    }

    public int getThreshold()        { return CLOSED_EYE_THRESHOLD; }
    public int getClosedFrameCount() { return closedFrameCount;      }
    public DriverStatus getStatus()  { return currentStatus;         }
}
