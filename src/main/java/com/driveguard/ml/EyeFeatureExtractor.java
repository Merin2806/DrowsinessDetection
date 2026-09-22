package com.driveguard.ml;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.HOGDescriptor;

import java.util.ArrayList;
import java.util.List;

/**
 * Extracts HOG (Histogram of Oriented Gradients) feature vectors from
 * single-eye images for use in SVM training and inference.
 *
 * HOG Configuration:
 *   Image size  : 64 × 32 pixels
 *   Cell size   : 8 × 8 pixels
 *   Block size  : 16 × 16 pixels (2 × 2 cells)
 *   Block stride: 8 × 8 pixels  (50% overlap)
 *   Histogram bins: 9
 *
 * Feature vector length = (blockCols * blockRows) * 4 cells/block * 9 bins
 *   where blockCols = (imgW - blockW) / strideW + 1 = (64-16)/8 + 1 = 7
 *         blockRows = (imgH - blockH) / strideH + 1 = (32-16)/8 + 1 = 3
 *   → 7 × 3 × 4 × 9 = 756 floats per image
 *
 * NOTE: The actual length is computed by OpenCV at runtime and can be
 * retrieved via getFeatureLength() after the first call to extract().
 */
public class EyeFeatureExtractor {

    // ── Default HOG parameters ──────────────────────────────────────────────
    public static final int IMG_WIDTH  = 64;
    public static final int IMG_HEIGHT = 32;

    private final int width;
    private final int height;
    private final boolean equalizeHist;

    private final Size winSize;
    private final Size blockSize;
    private final Size blockStride;
    private final Size cellSize;
    private final int  nbins;

    private final HOGDescriptor hog;
    private int featureLength = -1;

    public EyeFeatureExtractor() {
        this(IMG_WIDTH, IMG_HEIGHT, false);
    }

    public EyeFeatureExtractor(int width, int height) {
        this(width, height, false);
    }

    public EyeFeatureExtractor(int width, int height, boolean equalizeHist) {
        this.width = width;
        this.height = height;
        this.equalizeHist = equalizeHist;

        this.winSize = new Size(width, height);
        this.blockSize = new Size(16, 16);
        this.blockStride = new Size(8, 8);
        this.cellSize = new Size(8, 8);
        this.nbins = 9;

        this.hog = new HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins);
    }

    /**
     * Extract HOG feature vector from a single eye image file.
     *
     * @param imagePath absolute path to the image file
     * @return a 1 × N Mat of CV_32F floats (one row = one sample),
     *         or null if the image could not be read
     */
    public Mat extract(String imagePath) {
        Mat img = Imgcodecs.imread(imagePath);
        if (img.empty()) {
            System.err.println("[EyeFeatureExtractor] Cannot read: " + imagePath);
            return null;
        }

        try {
            return extract(img);
        } finally {
            img.release();
        }
    }

    /**
     * Extract HOG feature vector directly from an OpenCV Mat image.
     */
    public Mat extract(Mat img) {
        if (img == null || img.empty()) return null;

        // Convert to grayscale
        Mat gray = new Mat();
        if (img.channels() == 3) {
            Imgproc.cvtColor(img, gray, Imgproc.COLOR_BGR2GRAY);
        } else if (img.channels() == 4) {
            Imgproc.cvtColor(img, gray, Imgproc.COLOR_BGRA2GRAY);
        } else {
            gray = img.clone();
        }

        // Optional histogram equalization to normalize illumination
        if (equalizeHist) {
            Mat eq = new Mat();
            Imgproc.equalizeHist(gray, eq);
            gray.release();
            gray = eq;
        }

        // Resize to fixed window size
        Mat resized = new Mat();
        Imgproc.resize(gray, resized, winSize);

        // Compute HOG descriptor
        MatOfFloat descriptors = new MatOfFloat();
        hog.compute(resized, descriptors);

        // descriptors is a column vector (N × 1); convert to row vector (1 × N)
        Mat rowVec = descriptors.reshape(1, 1);

        if (featureLength < 0) {
            featureLength = (int) rowVec.total();
        }

        Mat result = new Mat();
        rowVec.convertTo(result, CvType.CV_32F);

        gray.release();
        resized.release();
        descriptors.release();

        return result;
    }

    public int getFeatureLength() {
        if (featureLength < 0) {
            featureLength = computeExpectedFeatureLength(width, height);
        }
        return featureLength;
    }

    public int getWidth()  { return width; }
    public int getHeight() { return height; }
    public boolean isHistogramEqualized() { return equalizeHist; }

    public static int getImgWidth()  { return IMG_WIDTH;  }
    public static int getImgHeight() { return IMG_HEIGHT; }
    public static int getNBins()     { return 9; }
    public static int getCellSize()  { return 8; }

    public static int computeExpectedFeatureLength() {
        return computeExpectedFeatureLength(IMG_WIDTH, IMG_HEIGHT);
    }

    public static int computeExpectedFeatureLength(int w, int h) {
        int blockCols = (w - 16) / 8 + 1;
        int blockRows = (h - 16) / 8 + 1;
        int cellsPerBlock = (16 / 8) * (16 / 8); // 4
        return blockCols * blockRows * cellsPerBlock * 9;
    }

    public List<String> getConfigLines() {
        List<String> lines = new ArrayList<>();
        lines.add("# HOG feature extraction configuration");
        lines.add("# These values MUST match between training and inference");
        lines.add("img.width="   + width);
        lines.add("img.height="  + height);
        lines.add("hog.cell.size=8");
        lines.add("hog.block.size=16");
        lines.add("hog.block.stride=8");
        lines.add("hog.bins=9");
        lines.add("hog.equalize.hist=" + equalizeHist);
        lines.add("label.awake=0");
        lines.add("label.sleepy=1");
        return lines;
    }
}
