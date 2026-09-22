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
    /** Running count of consecutive frames where eyes were sleepy or not detected. */
    private final int closedFrameCount;
    private final com.driveguard.ml.EyeSvmClassifier.Prediction aiPrediction;
    private final com.driveguard.ml.EyeSvmClassifier.Prediction leftEyePrediction;
    private final com.driveguard.ml.EyeSvmClassifier.Prediction rightEyePrediction;
    private final int eyeCount;
    private final int sleepyThreshold;

    // ── Constructors ──────────────────────────────────────────────────────

    public DetectionResult(boolean faceDetected,
                           EyeStatus eyeStatus,
                           DriverStatus driverStatus,
                           int closedFrameCount) {
        this(faceDetected, eyeStatus, driverStatus, closedFrameCount,
             com.driveguard.ml.EyeSvmClassifier.Prediction.NO_EYE,
             com.driveguard.ml.EyeSvmClassifier.Prediction.NO_EYE,
             com.driveguard.ml.EyeSvmClassifier.Prediction.NO_EYE,
             (eyeStatus == EyeStatus.OPEN ? 1 : 0), 20);
    }

    public DetectionResult(boolean faceDetected,
                           EyeStatus eyeStatus,
                           DriverStatus driverStatus,
                           int closedFrameCount,
                           com.driveguard.ml.EyeSvmClassifier.Prediction aiPrediction,
                           int eyeCount,
                           int sleepyThreshold) {
        this(faceDetected, eyeStatus, driverStatus, closedFrameCount,
             aiPrediction, aiPrediction, aiPrediction,
             eyeCount, sleepyThreshold);
    }

    public DetectionResult(boolean faceDetected,
                           EyeStatus eyeStatus,
                           DriverStatus driverStatus,
                           int closedFrameCount,
                           com.driveguard.ml.EyeSvmClassifier.Prediction aiPrediction,
                           com.driveguard.ml.EyeSvmClassifier.Prediction leftEyePrediction,
                           com.driveguard.ml.EyeSvmClassifier.Prediction rightEyePrediction,
                           int eyeCount,
                           int sleepyThreshold) {
        this.faceDetected        = faceDetected;
        this.eyeStatus           = eyeStatus;
        this.driverStatus        = driverStatus;
        this.closedFrameCount    = closedFrameCount;
        this.aiPrediction        = aiPrediction != null ? aiPrediction : com.driveguard.ml.EyeSvmClassifier.Prediction.NO_EYE;
        this.leftEyePrediction   = leftEyePrediction != null ? leftEyePrediction : com.driveguard.ml.EyeSvmClassifier.Prediction.NO_EYE;
        this.rightEyePrediction  = rightEyePrediction != null ? rightEyePrediction : com.driveguard.ml.EyeSvmClassifier.Prediction.NO_EYE;
        this.eyeCount            = eyeCount;
        this.sleepyThreshold     = sleepyThreshold;
    }

    // ── Accessors ─────────────────────────────────────────────────────────

    public boolean isFaceDetected()                                                 { return faceDetected;        }
    public EyeStatus getEyeStatus()                                                 { return eyeStatus;           }
    public DriverStatus getDriverStatus()                                           { return driverStatus;        }
    public int getClosedFrameCount()                                                { return closedFrameCount;    }
    public com.driveguard.ml.EyeSvmClassifier.Prediction getAiPrediction()         { return aiPrediction;        }
    public com.driveguard.ml.EyeSvmClassifier.Prediction getLeftEyePrediction()     { return leftEyePrediction;   }
    public com.driveguard.ml.EyeSvmClassifier.Prediction getRightEyePrediction()    { return rightEyePrediction;  }
    public int getEyeCount()                                                        { return eyeCount;            }
    public int getSleepyThreshold()                                                 { return sleepyThreshold;     }

    @Override
    public String toString() {
        return "DetectionResult{face=" + faceDetected
                + ", eyes=" + eyeStatus
                + ", eyeCount=" + eyeCount
                + ", ai=" + aiPrediction
                + ", leftAi=" + leftEyePrediction
                + ", rightAi=" + rightEyePrediction
                + ", status=" + driverStatus
                + ", closedFrames=" + closedFrameCount + "/" + sleepyThreshold + "}";
    }
}
