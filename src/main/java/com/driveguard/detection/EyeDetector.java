package com.driveguard.detection;

import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.objdetect.CascadeClassifier;

import java.io.IOException;
import java.nio.file.Path;

/**
 * Detects eyes inside a face region of interest (ROI) using the Haar eye cascade.
 *
 * Input is the sub-Mat cropped to the face rectangle (upper portion only).
 * Returned rectangles are relative to that sub-Mat, not the full frame.
 */
public class EyeDetector {

    private final CascadeClassifier classifier;

    public EyeDetector() throws IOException {
        classifier = new CascadeClassifier();
        Path cascadePath = FaceDetector.extractCascadeToTemp("/cascades/haarcascade_eye.xml");
        if (!classifier.load(cascadePath.toString())) {
            throw new IOException(
                "CascadeClassifier.load() returned false for: " + cascadePath + "\n" +
                "This usually means the XML file is corrupt or empty.");
        }
    }

    /**
     * Detect eyes inside the given face ROI.
     *
     * @param faceRoi grayscale sub-Mat corresponding to the upper part of the face
     * @return array of eye bounding rectangles relative to faceRoi; empty if none found
     */
    public Rect[] detect(Mat faceRoi) {
        MatOfRect eyes = new MatOfRect();
        /*
         * detectMultiScale parameters:
         *   scaleFactor  = 1.1
         *   minNeighbors = 15   — increased from 3 to 15 to STRICTLY require high confidence.
         *                         This prevents closed eyelids and eyebrows from being falsely 
         *                         detected as open eyes.
         *   flags        = 0
         *   minSize      = 20×20
         *   maxSize      = new Size()
         */
        classifier.detectMultiScale(faceRoi, eyes, 1.1, 15, 0,
                new Size(20, 20), new Size());
        return eyes.toArray();
    }
}
