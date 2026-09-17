package com.driveguard.model;

/**
 * Immutable value object representing the detection result for one video frame.
 * Passed from the camera thread to the UI layer on each processed frame.
 */
public class DetectionResult {

    // ── Inner enums ────────────────────────────────────────────────────────

    public enum DriverStatus {
        /** Face detected and eyes are open (or within closed-eye tolerance window). */
        ACTIVE,
        /** Face detected, eyes closed for more than CLOSED_EYE_THRESHOLD frames. */
        DROWSY,
        /** No face detected — cannot make a drowsiness determination. */
        WAITING
    }

    public enum EyeStatus {
        OPEN,
        CLOSED,
        NOT_DETECTED   // no face detected at all
    }

    // ── Fields ────────────────────────────────────────────────────────────

    private final boolean faceDetected;
    private final EyeStatus eyeStatus;
    private final DriverStatus driverStatus;
    /** Running count of consecutive frames where eyes were not detected. */
    private final int closedFrameCount;

    // ── Constructor ───────────────────────────────────────────────────────

    public DetectionResult(boolean faceDetected,
                           EyeStatus eyeStatus,
                           DriverStatus driverStatus,
                           int closedFrameCount) {
        this.faceDetected     = faceDetected;
        this.eyeStatus        = eyeStatus;
        this.driverStatus     = driverStatus;
        this.closedFrameCount = closedFrameCount;
    }

    // ── Accessors ─────────────────────────────────────────────────────────

    public boolean isFaceDetected()    { return faceDetected;     }
    public EyeStatus getEyeStatus()    { return eyeStatus;        }
    public DriverStatus getDriverStatus() { return driverStatus;  }
    public int getClosedFrameCount()   { return closedFrameCount; }

    @Override
    public String toString() {
        return "DetectionResult{face=" + faceDetected
                + ", eyes=" + eyeStatus
                + ", status=" + driverStatus
                + ", closedFrames=" + closedFrameCount + "}";
    }
}
