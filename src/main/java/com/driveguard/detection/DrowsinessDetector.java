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

    // ── CONFIGURABLE THRESHOLD — consecutive sleepy frames to trigger drowsiness ──
    public static final int SLEEPY_FRAME_THRESHOLD = 20;
    // ─────────────────────────────────────────────────────────────────────────

    private int closedFrameCount = 0;
    private DriverStatus currentStatus = DriverStatus.WAITING;

    /**
     * Process the detection and dual-eye AI classification result for one video frame.
     *
     * @param faceDetected true if a face was found in this frame
     * @param haarEyeCount number of eye ROIs detected by Haar cascade (for diagnostics only)
     * @param overallAi AI classification combined from both eyes
     * @param leftAi AI classification of left fixed eye ROI
     * @param rightAi AI classification of right fixed eye ROI
     * @return a DetectionResult capturing the updated state
     */
    public DetectionResult process(boolean faceDetected,
                                   int haarEyeCount,
                                   com.driveguard.ml.EyeSvmClassifier.Prediction overallAi,
                                   com.driveguard.ml.EyeSvmClassifier.Prediction leftAi,
                                   com.driveguard.ml.EyeSvmClassifier.Prediction rightAi) {

        // ── No face detected ────────────────────────────────────────────────
        if (!faceDetected) {
            closedFrameCount = 0;
            currentStatus    = DriverStatus.WAITING;
            return new DetectionResult(false, EyeStatus.NOT_DETECTED, DriverStatus.WAITING, 0,
                    com.driveguard.ml.EyeSvmClassifier.Prediction.NO_EYE,
                    com.driveguard.ml.EyeSvmClassifier.Prediction.NO_EYE,
                    com.driveguard.ml.EyeSvmClassifier.Prediction.NO_EYE,
                    0, SLEEPY_FRAME_THRESHOLD);
        }

        EyeStatus eyeStatus;

        // ── Face detected: evaluate SVM classifications ─────────────────────
        // Decision is driven by AI SVM predictions on fixed eye ROIs, NOT Haar 0-eye count.
        if (leftAi == com.driveguard.ml.EyeSvmClassifier.Prediction.SLEEPY &&
            rightAi == com.driveguard.ml.EyeSvmClassifier.Prediction.SLEEPY) {
            // Both eyes confirmed sleepy/closed by SVM
            closedFrameCount++;
            eyeStatus = EyeStatus.CLOSED;

        } else if (leftAi == com.driveguard.ml.EyeSvmClassifier.Prediction.AWAKE &&
                   rightAi == com.driveguard.ml.EyeSvmClassifier.Prediction.AWAKE) {
            // Both eyes confirmed alert/open by SVM
            closedFrameCount = 0;
            eyeStatus = EyeStatus.OPEN;

        } else if ((leftAi == com.driveguard.ml.EyeSvmClassifier.Prediction.AWAKE && rightAi == com.driveguard.ml.EyeSvmClassifier.Prediction.SLEEPY) ||
                   (leftAi == com.driveguard.ml.EyeSvmClassifier.Prediction.SLEEPY && rightAi == com.driveguard.ml.EyeSvmClassifier.Prediction.AWAKE)) {
            // Conservative/temporal strategy:
            // One eye is AWAKE and one is SLEEPY.
            // A driver with visual awareness in at least one eye is NOT declared drowsy.
            // Do NOT increment closedFrameCount towards DROWSY.
            // If the counter was already non-zero (e.g. slight jitter during eye closure),
            // decay slowly by 1 rather than resetting to 0, but never increase.
            if (closedFrameCount > 0) {
                closedFrameCount--;
            }
            eyeStatus = EyeStatus.OPEN;

        } else {
            // Inconclusive or NO_EYE
            eyeStatus = EyeStatus.NOT_DETECTED;
        }

        // Evaluate against threshold
        if (closedFrameCount >= SLEEPY_FRAME_THRESHOLD) {
            currentStatus = DriverStatus.DROWSY;
        } else {
            currentStatus = DriverStatus.ACTIVE;
        }

        return new DetectionResult(true, eyeStatus, currentStatus, closedFrameCount,
                overallAi, leftAi, rightAi, haarEyeCount, SLEEPY_FRAME_THRESHOLD);
    }

    /**
     * Backward-compatible overload.
     */
    public DetectionResult process(boolean faceDetected,
                                   int eyeCount,
                                   com.driveguard.ml.EyeSvmClassifier.Prediction aiPrediction) {
        return process(faceDetected, eyeCount, aiPrediction, aiPrediction, aiPrediction);
    }

    /**
     * Backward-compatible overload for boolean eyesDetected.
     */
    public DetectionResult process(boolean faceDetected, boolean eyesDetected) {
        return process(faceDetected, eyesDetected ? 1 : 0,
                eyesDetected ? com.driveguard.ml.EyeSvmClassifier.Prediction.AWAKE
                             : com.driveguard.ml.EyeSvmClassifier.Prediction.NO_EYE);
    }

    /** Reset detector state (called when camera restarts). */
    public void reset() {
        closedFrameCount = 0;
        currentStatus    = DriverStatus.WAITING;
    }

    public int getThreshold()        { return SLEEPY_FRAME_THRESHOLD; }
    public int getClosedFrameCount() { return closedFrameCount;      }
    public DriverStatus getStatus()  { return currentStatus;         }
}
