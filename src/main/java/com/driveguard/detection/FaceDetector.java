package com.driveguard.detection;

import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.objdetect.CascadeClassifier;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

/**
 * Detects human faces in a video frame using the Haar frontal-face cascade.
 *
 * The XML cascade file is bundled inside the classpath (resources/cascades/).
 * Because CascadeClassifier.load() requires a real filesystem path, we first
 * extract the resource to a temporary file, then load it from there.
 */
public class FaceDetector {

    private final CascadeClassifier classifier;

    public FaceDetector() throws IOException {
        classifier = new CascadeClassifier();
        Path cascadePath = extractCascadeToTemp("/cascades/haarcascade_frontalface_default.xml");
        if (!classifier.load(cascadePath.toString())) {
            throw new IOException(
                "CascadeClassifier.load() returned false for: " + cascadePath + "\n" +
                "This usually means the XML file is corrupt or empty.");
        }
    }

    /**
     * Detect faces in the given (gray-scale or colour) frame.
     *
     * @param frame an OpenCV Mat (grayscale recommended for speed)
     * @return array of face bounding rectangles; empty array if none found
     */
    public Rect[] detect(Mat frame) {
        MatOfRect faces = new MatOfRect();
        /*
         * detectMultiScale parameters:
         *   scaleFactor  = 1.1  — image pyramid scale between levels
         *   minNeighbors = 4    — how many neighbours a candidate needs (higher = fewer false positives)
         *   flags        = 0    — legacy flag, not used in modern OpenCV
         *   minSize      = 60×60 — ignore tiny faces (far from camera)
         *   maxSize      = new Size() — no upper limit
         */
        classifier.detectMultiScale(frame, faces, 1.1, 4, 0,
                new Size(60, 60), new Size());
        return faces.toArray();
    }

    /**
     * Extract a classpath resource XML to a temp file so OpenCV can load it
     * by filesystem path. The temp file is scheduled for deletion on JVM exit.
     *
     * Package-private so EyeDetector can reuse it.
     */
    static Path extractCascadeToTemp(String resourcePath) throws IOException {
        try (InputStream is = FaceDetector.class.getResourceAsStream(resourcePath)) {
            if (is == null) {
                String filename = resourcePath.substring(resourcePath.lastIndexOf('/') + 1);
                throw new IOException(
                    "Haar cascade not found on classpath: " + resourcePath + "\n\n" +
                    "ACTION REQUIRED:\n" +
                    "  Download '" + filename + "' from:\n" +
                    "  https://github.com/opencv/opencv/tree/master/data/haarcascades\n\n" +
                    "  Place it at:\n" +
                    "  src/main/resources/cascades/" + filename);
            }
            // Create a temp file with a recognisable name for debugging
            String suffix = "_" + resourcePath.substring(resourcePath.lastIndexOf('/') + 1);
            Path temp = Files.createTempFile("driveguard_cascade", suffix);
            temp.toFile().deleteOnExit();
            Files.copy(is, temp, StandardCopyOption.REPLACE_EXISTING);
            return temp;
        }
    }
}
