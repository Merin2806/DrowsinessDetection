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
    private final DrowsinessDetector  drowsinessDetector;

    /** Called on the JavaFX Application Thread with (fxImage, result) per frame. */
    private final BiConsumer<Image, DetectionResult> frameCallback;

    // ── Thread control ────────────────────────────────────────────────────
    private VideoCapture capture;
    private Thread       captureThread;
    private final AtomicBoolean running = new AtomicBoolean(false);

    // ── Constructor ───────────────────────────────────────────────────────

    public CameraService(FaceDetector faceDetector,
                         EyeDetector eyeDetector,
                         DrowsinessDetector drowsinessDetector,
                         BiConsumer<Image, DetectionResult> frameCallback) {
        this.faceDetector       = faceDetector;
        this.eyeDetector        = eyeDetector;
        this.drowsinessDetector = drowsinessDetector;
        this.frameCallback      = frameCallback;
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

            // ── Pre-process for detection ─────────────────────────────────
            // Convert to grayscale (Haar cascade expects single-channel input)
            Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_BGR2GRAY);
            // Equalise histogram to improve detection under variable lighting
            Imgproc.equalizeHist(grayFrame, grayFrame);

            // ── Face Detection ────────────────────────────────────────────
            Rect[] faces     = faceDetector.detect(grayFrame);
            boolean faceFound = faces.length > 0;
            boolean eyesFound = false;
            int currentEyeCount = 0;

            if (faceFound) {
                // Use the largest detected face (driver is closest to camera)
                Rect face = largestRect(faces);

                // Draw green rectangle around the detected face
                Imgproc.rectangle(frame, face.tl(), face.br(), COLOUR_GREEN, LINE_THICKNESS);
                Imgproc.putText(frame, "FACE", new Point(face.x, face.y - 6),
                        Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, COLOUR_GREEN, 1);

                // ── Eye Detection inside the face region ──────────────────
                // Restrict to the upper 60% of the face — eyes live there.
                // This prevents the mouth/chin from generating false eye hits.
                int roiY      = face.y;
                int roiHeight = (int) (face.height * 0.60);

                // Guard against out-of-bounds (should not happen with valid face rects)
                roiHeight = Math.min(roiHeight, grayFrame.rows() - roiY);

                Rect  eyeSearchArea  = new Rect(face.x, roiY, face.width, roiHeight);
                Mat   faceGrayRoi    = grayFrame.submat(eyeSearchArea);
                Rect[] eyes          = eyeDetector.detect(faceGrayRoi);
                currentEyeCount      = eyes.length;

                // At least ONE eye detected → eyes are considered open
                if (currentEyeCount >= 1) {
                    eyesFound = true;
                    for (Rect eye : eyes) {
                        // Translate eye coords from face-ROI space to full-frame space
                        Point eyeTl = new Point(face.x + eye.x,           face.y + eye.y);
                        Point eyeBr = new Point(face.x + eye.x + eye.width, face.y + eye.y + eye.height);
                        Imgproc.rectangle(frame, eyeTl, eyeBr, COLOUR_YELLOW, LINE_THICKNESS);
                        Imgproc.putText(frame, "EYE", eyeTl,
                                Imgproc.FONT_HERSHEY_SIMPLEX, 0.4, COLOUR_YELLOW, 1);
                    }
                }

                faceGrayRoi.release();
            }

            // ── Drowsiness State Machine ──────────────────────────────────
            DetectionResult result = drowsinessDetector.process(faceFound, eyesFound);

            // ── Annotate frame with status overlay ────────────────────────
            annotateFrame(frame, result);

            // ── Draw Diagnostics ──────────────────────────────────────────
            String diag1 = "Eyes detected: " + currentEyeCount;
            String diag2 = "Closed frames: " + result.getClosedFrameCount() + "/" + drowsinessDetector.getThreshold();
            Imgproc.putText(frame, diag1, new Point(8, 120), Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, new Scalar(255, 255, 0), 2);
            Imgproc.putText(frame, diag2, new Point(8, 145), Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, new Scalar(255, 255, 0), 2);

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
