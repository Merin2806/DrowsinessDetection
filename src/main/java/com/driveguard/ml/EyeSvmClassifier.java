package com.driveguard.ml;

import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.ml.SVM;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Properties;

/**
 * Loads the trained OpenCV SVM model and HOG configuration, and classifies
 * cropped eye ROIs into AWAKE (0) or SLEEPY (1).
 *
 * Preprocessing is mathematically identical to training:
 *   1. Grayscale conversion
 *   2. Histogram Equalization (Imgproc.equalizeHist)
 *   3. Resize to 32x32 pixels (1:1 square aspect ratio)
 *   4. HOG descriptor computation (8x8 cells, 16x16 blocks, 9 bins -> 324 floats)
 *   5. Linear SVM prediction
 */
public class EyeSvmClassifier {

    public enum Prediction {
        AWAKE,
        SLEEPY,
        NO_EYE
    }

    private static final String DEFAULT_MODEL_PATH  = "model/eye_svm.yml";
    private static final String DEFAULT_CONFIG_PATH = "model/hog_config.properties";

    private final SVM svm;
    private final EyeFeatureExtractor extractor;

    public EyeSvmClassifier() throws IOException {
        this(DEFAULT_MODEL_PATH, DEFAULT_CONFIG_PATH);
    }

    public EyeSvmClassifier(String modelPath, String configPath) throws IOException {
        File mFile = new File(modelPath);
        if (!mFile.exists()) {
            throw new IOException("Trained SVM model not found at: " + mFile.getAbsolutePath() +
                    "\nPlease ensure model/eye_svm.yml exists before starting.");
        }

        // 1. Load HOG configuration
        int imgWidth = 32;
        int imgHeight = 32;
        boolean equalizeHist = true;

        File cFile = new File(configPath);
        if (cFile.exists()) {
            Properties props = new Properties();
            try (FileInputStream in = new FileInputStream(cFile)) {
                props.load(in);
                imgWidth = Integer.parseInt(props.getProperty("img.width", "32"));
                imgHeight = Integer.parseInt(props.getProperty("img.height", "32"));
                equalizeHist = Boolean.parseBoolean(props.getProperty("hog.equalize.hist", "true"));
            } catch (Exception e) {
                System.err.println("[WARN] Could not parse " + configPath + ", using defaults: " + e.getMessage());
            }
        }

        this.extractor = new EyeFeatureExtractor(imgWidth, imgHeight, equalizeHist);
        System.out.printf("[EyeSvmClassifier] Initialized feature extractor: %dx%d, equalizeHist=%b, featureLen=%d%n",
                imgWidth, imgHeight, equalizeHist, extractor.getFeatureLength());

        // 2. Load trained SVM model
        try {
            this.svm = SVM.load(modelPath);
            if (this.svm == null || this.svm.empty()) {
                throw new IOException("SVM.load returned null or empty for: " + modelPath);
            }
            System.out.printf("[EyeSvmClassifier] Loaded trained SVM model from: %s (Type: %d, Kernel: %d, C=%.2f)%n",
                    modelPath, svm.getType(), svm.getKernelType(), svm.getC());
        } catch (Exception e) {
            throw new IOException("Failed to load SVM model from " + modelPath + ": " + e.getMessage(), e);
        }
    }

    /**
     * Classifies a single cropped eye ROI Mat.
     *
     * @param eyeRoi cropped sub-Mat representing the eye
     * @return Prediction (AWAKE, SLEEPY, or NO_EYE)
     */
    public Prediction predict(Mat eyeRoi) {
        if (eyeRoi == null || eyeRoi.empty() || eyeRoi.cols() < 8 || eyeRoi.rows() < 8) {
            return Prediction.NO_EYE;
        }

        Mat features = extractor.extract(eyeRoi);
        if (features == null || features.empty()) {
            return Prediction.NO_EYE;
        }

        try {
            float label = svm.predict(features);
            int labelInt = Math.round(label);
            return (labelInt == 1) ? Prediction.SLEEPY : Prediction.AWAKE;
        } finally {
            features.release();
        }
    }

    /**
     * Safely crop an eye rectangle from a parent Mat and classify it.
     */
    public Prediction predict(Mat parent, Rect eyeRect) {
        if (parent == null || parent.empty() || eyeRect == null || eyeRect.width < 8 || eyeRect.height < 8) {
            return Prediction.NO_EYE;
        }

        // Clamp rectangle to parent bounds
        int x = Math.max(0, eyeRect.x);
        int y = Math.max(0, eyeRect.y);
        int w = Math.min(eyeRect.width, parent.cols() - x);
        int h = Math.min(eyeRect.height, parent.rows() - y);

        if (w < 8 || h < 8) return Prediction.NO_EYE;

        Rect safeRect = new Rect(x, y, w, h);
        Mat eyeRoi = parent.submat(safeRect);
        try {
            return predict(eyeRoi);
        } finally {
            eyeRoi.release();
        }
    }

    public EyeFeatureExtractor getExtractor() {
        return extractor;
    }

    public static void main(String[] args) throws Exception {
        nu.pattern.OpenCV.loadLocally();
        System.out.println("Testing EyeSvmClassifier model loading...");
        EyeSvmClassifier classifier = new EyeSvmClassifier();
        System.out.println("SUCCESS: EyeSvmClassifier loaded successfully.");
        System.out.println("Feature vector length: " + classifier.getExtractor().getFeatureLength());

        // Test with a dummy 32x32 image
        Mat dummy = new Mat(32, 32, org.opencv.core.CvType.CV_8UC1, new org.opencv.core.Scalar(128));
        Prediction pred = classifier.predict(dummy);
        dummy.release();
        System.out.println("Dummy inference result: " + pred);
        System.out.println("ALL INFERENCE CHECKS PASSED.");
    }
}
