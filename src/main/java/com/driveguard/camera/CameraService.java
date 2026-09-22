package com.driveguard.camera;

import com.driveguard.detection.DrowsinessDetector;
import com.driveguard.detection.EyeDetector;
import com.driveguard.detection.FaceDetector;
import com.driveguard.model.DetectionResult;

import javafx.application.Platform;
import javafx.scene.image.Image;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

import java.io.ByteArrayInputStream;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.BiConsumer;

/**
 * Manages the OpenCV VideoCapture loop on a dedicated background thread.
 *
 * For each frame it:
 *   1. Reads the raw frame from the webcam
 *   2. Converts to grayscale + equalises histogram (improves detection accuracy)
 *   3. Runs face detection
 *   4. If face found, runs eye detection on the upper-60% face ROI
 *   5. Calls DrowsinessDetector to update state
 *   6. Annotates the colour frame with bounding boxes and status text
 *   7. Converts Mat → JavaFX Image and pushes to the UI via Platform.runLater()
 */
public class CameraService {

    // ── Annotation colours (OpenCV uses BGR, not RGB) ──────────────────────
    private static final Scalar COLOUR_GREEN  = new Scalar(0,   220, 100);  // face box
    private static final Scalar COLOUR_YELLOW = new Scalar(0,   200, 230);  // eye box
    private static final Scalar COLOUR_RED    = new Scalar(30,  30,  240);  // alert text
    private static final int    LINE_THICKNESS = 2;

    // ── Services ──────────────────────────────────────────────────────────
    private final FaceDetector        faceDetector;
    private final EyeDetector         eyeDetector;
    private final com.driveguard.ml.EyeSvmClassifier eyeSvmClassifier;
    private final DrowsinessDetector  drowsinessDetector;

    /** Called on the JavaFX Application Thread with (fxImage, result) per frame. */
    private final BiConsumer<Image, DetectionResult> frameCallback;

    // ── Thread control ────────────────────────────────────────────────────
    private VideoCapture capture;
    private Thread       captureThread;
    private final AtomicBoolean running = new AtomicBoolean(false);

    // ── Constructors ──────────────────────────────────────────────────────

    public CameraService(FaceDetector faceDetector,
                         EyeDetector eyeDetector,
                         com.driveguard.ml.EyeSvmClassifier eyeSvmClassifier,
                         DrowsinessDetector drowsinessDetector,
                         BiConsumer<Image, DetectionResult> frameCallback) {
        this.faceDetector       = faceDetector;
        this.eyeDetector        = eyeDetector;
        this.eyeSvmClassifier   = eyeSvmClassifier;
        this.drowsinessDetector = drowsinessDetector;
        this.frameCallback      = frameCallback;
    }

    public CameraService(FaceDetector faceDetector,
                         EyeDetector eyeDetector,
                         DrowsinessDetector drowsinessDetector,
                         BiConsumer<Image, DetectionResult> frameCallback) {
        this(faceDetector, eyeDetector, null, drowsinessDetector, frameCallback);
    }

    // ── Public API ────────────────────────────────────────────────────────

    /**
     * Open webcam index 0 and start the processing loop.
     *
     * @throws IllegalStateException if the webcam cannot be opened
     */
    public void start() {
        if (running.get()) return;

        capture = new VideoCapture(0);
        if (!capture.isOpened()) {
            throw new IllegalStateException(
                "Cannot open webcam (index 0).\n\n" +
                "Possible causes:\n" +
                "  • No webcam is connected\n" +
                "  • Another application is already using the camera\n" +
                "  • Driver or permission issue\n\n" +
                "Please close other apps that may be using the camera and try again.");
        }

        drowsinessDetector.reset();
        running.set(true);

        captureThread = new Thread(this::captureLoop, "DriveGuard-CameraThread");
        captureThread.setDaemon(true); // JVM will not wait for this thread on exit
        captureThread.start();
    }

    /**
     * Stop the capture loop and release the VideoCapture resource.
     * Blocks until the background thread exits (max 2 s).
     */
    public void stop() {
        running.set(false);
        if (captureThread != null) {
            try {
                captureThread.join(2000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            captureThread = null;
        }
        releaseCapture();
    }

    public boolean isRunning() {
        return running.get();
    }

    // ── Main camera loop ──────────────────────────────────────────────────

    private void captureLoop() {
        Mat frame     = new Mat();
        Mat grayFrame = new Mat();

        while (running.get()) {
            // ── Read frame ────────────────────────────────────────────────
            if (!capture.read(frame) || frame.empty()) {
                System.err.println("[CameraService] Empty frame — skipping.");
                sleepMs(50);
                continue;
            }

            // ── Pre-process for face detection ────────────────────────────
            Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_BGR2GRAY);
            Imgproc.equalizeHist(grayFrame, grayFrame);

            // ── Face Detection ────────────────────────────────────────────
            Rect[] faces = faceDetector.detect(grayFrame);
            boolean faceFound = faces.length > 0;
            int currentEyeCount = 0;
            DetectionResult result;

            if (faceFound) {
                // Use largest detected face (driver is closest to camera)
                Rect face = largestRect(faces);

                // ── Anthropometric Fixed Eye ROI Estimation ───────────────────
                // Calculate expected left and right eye locations based on facial geometry:
                //   • Vertical: 23% to 51% from the top of the face box (height: 28%)
                //   • Screen-Left Eye (Subject Right): 15% to 45% of face width (width: 30%)
                //   • Screen-Right Eye (Subject Left): 55% to 85% of face width (width: 30%)
                // This produces ~1:1 square ROIs that encompass eyelids whether open or closed.
                int leftX  = face.x + (int) (face.width * 0.15);
                int leftY  = face.y + (int) (face.height * 0.23);
                int leftW  = (int) (face.width * 0.30);
                int leftH  = (int) (face.height * 0.28);

                int rightX = face.x + (int) (face.width * 0.55);
                int rightY = face.y + (int) (face.height * 0.23);
                int rightW = (int) (face.width * 0.30);
                int rightH = (int) (face.height * 0.28);

                Rect leftEyeRoi  = clampRect(new Rect(leftX, leftY, leftW, leftH), frame.cols(), frame.rows());
                Rect rightEyeRoi = clampRect(new Rect(rightX, rightY, rightW, rightH), frame.cols(), frame.rows());

                // ── AI SVM Inference on Fixed ROIs ────────────────────────────
                // Crop ROIs directly from unannotated BGR frame before drawing any boxes.
                // Preprocessing (Grayscale -> EqualizeHist -> Resize 32x32 -> HOG 324)
                // is executed identically to training.
                com.driveguard.ml.EyeSvmClassifier.Prediction leftEyePred  = com.driveguard.ml.EyeSvmClassifier.Prediction.NO_EYE;
                com.driveguard.ml.EyeSvmClassifier.Prediction rightEyePred = com.driveguard.ml.EyeSvmClassifier.Prediction.NO_EYE;

                if (leftEyeRoi != null && leftEyeRoi.width >= 8 && leftEyeRoi.height >= 8) {
                    Mat leftMat = frame.submat(leftEyeRoi);
                    if (eyeSvmClassifier != null) {
                        leftEyePred = eyeSvmClassifier.predict(leftMat);
                    }
                    leftMat.release();
                }

                if (rightEyeRoi != null && rightEyeRoi.width >= 8 && rightEyeRoi.height >= 8) {
                    Mat rightMat = frame.submat(rightEyeRoi);
                    if (eyeSvmClassifier != null) {
                        rightEyePred = eyeSvmClassifier.predict(rightMat);
                    }
                    rightMat.release();
                }

                // ── Combine Eye Predictions ───────────────────────────────────
                //   • Both AWAKE -> AWAKE
                //   • Both SLEEPY -> SLEEPY
                //   • One AWAKE & one SLEEPY -> Conservative strategy (AWAKE) to avoid false alarms
                com.driveguard.ml.EyeSvmClassifier.Prediction overallAiPred;
                if (leftEyePred == com.driveguard.ml.EyeSvmClassifier.Prediction.SLEEPY &&
                    rightEyePred == com.driveguard.ml.EyeSvmClassifier.Prediction.SLEEPY) {
                    overallAiPred = com.driveguard.ml.EyeSvmClassifier.Prediction.SLEEPY;
                } else if (leftEyePred == com.driveguard.ml.EyeSvmClassifier.Prediction.AWAKE ||
                           rightEyePred == com.driveguard.ml.EyeSvmClassifier.Prediction.AWAKE) {
                    overallAiPred = com.driveguard.ml.EyeSvmClassifier.Prediction.AWAKE;
                } else {
                    overallAiPred = com.driveguard.ml.EyeSvmClassifier.Prediction.NO_EYE;
                }

                // ── Haar Eye Detection (For Debugging / Visualization) ────────
                // Run on upper 60% of face; failure to detect does NOT block SVM decision.
                int roiY      = face.y;
                int roiHeight = (int) (face.height * 0.60);
                roiHeight     = Math.min(roiHeight, grayFrame.rows() - roiY);
                Rect eyeSearchArea = new Rect(face.x, roiY, face.width, roiHeight);
                Mat faceGrayRoi    = grayFrame.submat(eyeSearchArea);
                Rect[] haarEyes    = eyeDetector.detect(faceGrayRoi);
                currentEyeCount    = haarEyes.length;
                faceGrayRoi.release();

                // ── Update Drowsiness State Machine ───────────────────────────
                result = drowsinessDetector.process(faceFound, currentEyeCount, overallAiPred, leftEyePred, rightEyePred);

                // ── Draw Annotations on Video Frame ───────────────────────────
                // 1. Face box
                Imgproc.rectangle(frame, face.tl(), face.br(), COLOUR_GREEN, LINE_THICKNESS);
                Imgproc.putText(frame, "FACE", new Point(face.x, Math.max(16, face.y - 6)),
                        Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, COLOUR_GREEN, 1);

                // 2. Fixed Eye ROIs with AI Labels
                if (leftEyeRoi != null) {
                    Scalar leftColor = (leftEyePred == com.driveguard.ml.EyeSvmClassifier.Prediction.SLEEPY)
                            ? COLOUR_RED : COLOUR_GREEN;
                    Imgproc.rectangle(frame, leftEyeRoi.tl(), leftEyeRoi.br(), leftColor, LINE_THICKNESS);
                    Imgproc.putText(frame, "L: " + leftEyePred.name(),
                            new Point(leftEyeRoi.x, Math.max(14, leftEyeRoi.y - 4)),
                            Imgproc.FONT_HERSHEY_SIMPLEX, 0.45, leftColor, 1);
                }

                if (rightEyeRoi != null) {
                    Scalar rightColor = (rightEyePred == com.driveguard.ml.EyeSvmClassifier.Prediction.SLEEPY)
                            ? COLOUR_RED : COLOUR_GREEN;
                    Imgproc.rectangle(frame, rightEyeRoi.tl(), rightEyeRoi.br(), rightColor, LINE_THICKNESS);
                    Imgproc.putText(frame, "R: " + rightEyePred.name(),
                            new Point(rightEyeRoi.x, Math.max(14, rightEyeRoi.y - 4)),
                            Imgproc.FONT_HERSHEY_SIMPLEX, 0.45, rightColor, 1);
                }

                // 3. Optional Haar detected eye boxes in cyan (debug)
                for (Rect hEye : haarEyes) {
                    Point hTl = new Point(face.x + hEye.x, roiY + hEye.y);
                    Point hBr = new Point(face.x + hEye.x + hEye.width, roiY + hEye.y + hEye.height);
                    Imgproc.rectangle(frame, hTl, hBr, new Scalar(255, 200, 0), 1);
                }

            } else {
                // No face detected
                result = drowsinessDetector.process(false, 0,
                        com.driveguard.ml.EyeSvmClassifier.Prediction.NO_EYE,
                        com.driveguard.ml.EyeSvmClassifier.Prediction.NO_EYE,
                        com.driveguard.ml.EyeSvmClassifier.Prediction.NO_EYE);
            }

            // ── Annotate frame with status overlay ────────────────────────
            annotateFrame(frame, result);

            // ── Draw Diagnostics ──────────────────────────────────────────
            Scalar leftColor = (result.getLeftEyePrediction() == com.driveguard.ml.EyeSvmClassifier.Prediction.SLEEPY)
                    ? COLOUR_RED : COLOUR_GREEN;
            Scalar rightColor = (result.getRightEyePrediction() == com.driveguard.ml.EyeSvmClassifier.Prediction.SLEEPY)
                    ? COLOUR_RED : COLOUR_GREEN;
            Scalar overallColor = switch (result.getAiPrediction()) {
                case AWAKE   -> COLOUR_GREEN;
                case SLEEPY  -> COLOUR_RED;
                case NO_EYE  -> COLOUR_YELLOW;
            };

            Imgproc.putText(frame, "Left Eye AI: "  + result.getLeftEyePrediction().name(),
                    new Point(8, 115), Imgproc.FONT_HERSHEY_SIMPLEX, 0.50, leftColor, 2);
            Imgproc.putText(frame, "Right Eye AI: " + result.getRightEyePrediction().name(),
                    new Point(8, 138), Imgproc.FONT_HERSHEY_SIMPLEX, 0.50, rightColor, 2);
            Imgproc.putText(frame, "Overall AI: "   + result.getAiPrediction().name(),
                    new Point(8, 161), Imgproc.FONT_HERSHEY_SIMPLEX, 0.55, overallColor, 2);
            Imgproc.putText(frame, "Sleepy Frames: " + result.getClosedFrameCount() + "/" + drowsinessDetector.getThreshold(),
                    new Point(8, 184), Imgproc.FONT_HERSHEY_SIMPLEX, 0.50, new Scalar(255, 255, 0), 2);
            Imgproc.putText(frame, "Haar Eyes: " + currentEyeCount,
                    new Point(8, 207), Imgproc.FONT_HERSHEY_SIMPLEX, 0.45, new Scalar(200, 200, 200), 1);

            // ── Convert Mat (BGR) → JavaFX Image ─────────────────────────
            Image fxImage = matToFxImage(frame);

            // ── Push to UI on the JavaFX Application Thread ───────────────
            if (fxImage != null) {
                DetectionResult resultSnapshot = result; // effectively final for lambda
                Platform.runLater(() -> frameCallback.accept(fxImage, resultSnapshot));
            }

            // Target ~30 FPS
            sleepMs(33);
        }

        // Release Mats when loop exits
        frame.release();
        grayFrame.release();
    }

    // ── Frame annotation ─────────────────────────────────────────────────

    /**
     * Draw a status overlay directly onto the OpenCV colour frame.
     * This text appears inside the camera panel, not in the JavaFX status cards.
     */
    private void annotateFrame(Mat frame, DetectionResult result) {
        String text;
        Scalar colour;

        switch (result.getDriverStatus()) {
            case DROWSY -> {
                text   = "! DROWSY !";
                colour = COLOUR_RED;
                // Large bold-style alert text in the upper-left corner
                Imgproc.putText(frame, "DROWSINESS ALERT",
                        new Point(8, 68), Imgproc.FONT_HERSHEY_SIMPLEX, 0.9, COLOUR_RED, 2);
            }
            case ACTIVE  -> { text = "ACTIVE";   colour = COLOUR_GREEN;  }
            case WAITING -> { text = "NO FACE";  colour = COLOUR_YELLOW; }
            default      -> { text = "";          colour = COLOUR_GREEN;  }
        }

        Imgproc.putText(frame, "STATUS: " + text,
                new Point(8, 30), Imgproc.FONT_HERSHEY_SIMPLEX, 0.75, colour, 2);
    }

    // ── Mat → JavaFX Image conversion ────────────────────────────────────

    /**
     * Encode an OpenCV Mat to JPEG bytes, then wrap in a JavaFX Image.
     * Using JPEG keeps per-frame memory low at acceptable quality.
     */
    private Image matToFxImage(Mat mat) {
        try {
            MatOfByte mob = new MatOfByte();
            Imgcodecs.imencode(".jpg", mat, mob);
            byte[] bytes = mob.toArray();
            mob.release();
            return new Image(new ByteArrayInputStream(bytes));
        } catch (Exception e) {
            System.err.println("[CameraService] Frame conversion error: " + e.getMessage());
            return null;
        }
    }

    // ── Utilities ─────────────────────────────────────────────────────────

    /** Return the rectangle with the largest area. */
    private static Rect largestRect(Rect[] rects) {
        Rect largest = rects[0];
        for (Rect r : rects) {
            if (r.area() > largest.area()) largest = r;
        }
        return largest;
    }

    /** Ensure rectangle stays strictly within frame boundaries. */
    private static Rect clampRect(Rect r, int maxW, int maxH) {
        int x = Math.max(0, r.x);
        int y = Math.max(0, r.y);
        int w = Math.min(r.width, maxW - x);
        int h = Math.min(r.height, maxH - y);
        if (w <= 0 || h <= 0) return null;
        return new Rect(x, y, w, h);
    }

    private static void sleepMs(long ms) {
        try {
            Thread.sleep(ms);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    private void releaseCapture() {
        if (capture != null && capture.isOpened()) {
            capture.release();
        }
    }
}
