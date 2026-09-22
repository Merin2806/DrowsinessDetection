package com.driveguard.ml;

import nu.pattern.OpenCV;
import org.opencv.core.*;
import org.opencv.ml.Ml;
import org.opencv.ml.ParamGrid;
import org.opencv.ml.SVM;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

/**
 * Trains an OpenCV SVM classifier on HOG eye features extracted from
 * the MRL Eye Dataset.
 *
 * ── Label convention ──────────────────────────────────────────────────────
 *   0 = AWAKE   (folder: awake/)
 *   1 = SLEEPY  (folder: sleepy/)
 * ──────────────────────────────────────────────────────────────────────────
 *
 * Usage (via Maven):
 *   mvn exec:java -Dexec.mainClass="com.driveguard.ml.EyeSvmTrainer" \
 *                 -Dexec.args="C:\path\to\data" \
 *                 -Dexec.classpathScope=compile
 *
 * The argument must be the directory that contains train/, val/, test/.
 *
 * Output:
 *   model/eye_svm.yml            — trained SVM (OpenCV YAML)
 *   model/hog_config.properties  — HOG parameters for inference
 */
public class EyeSvmTrainer {

    // ── Label constants ────────────────────────────────────────────────────
    private static final int LABEL_AWAKE  = 0;
    private static final int LABEL_SLEEPY = 1;

    // ── Model output paths ─────────────────────────────────────────────────
    private static final String MODEL_DIR        = "model";
    private static final String MODEL_FILE       = MODEL_DIR + "/eye_svm.yml";
    private static final String CONFIG_FILE      = MODEL_DIR + "/hog_config.properties";

    // ── Supported image extensions ─────────────────────────────────────────
    private static final String[] SUPPORTED_EXTS = { ".png", ".jpg", ".jpeg", ".bmp", ".pgm" };

    // ── Entry point ────────────────────────────────────────────────────────

    public static void main(String[] args) throws IOException {
        // Load OpenCV native library (same pattern as main app)
        try {
            OpenCV.loadLocally();
        } catch (Exception e) {
            System.err.println("[ERROR] Failed to load OpenCV native library: " + e.getMessage());
            System.exit(1);
        }

        String dataRoot = null;
        if (args.length >= 1 && !args[0].trim().isEmpty()) {
            dataRoot = args[0].trim();
        } else {
            dataRoot = System.getProperty("datasetPath");
        }

        if (dataRoot == null || dataRoot.trim().isEmpty()) {
            System.err.println("Usage: EyeSvmTrainer <dataset-root> [max-samples-per-class]");
            System.err.println("  or via: -DdatasetPath=\"C:\\path\\to\\data\" [-DmaxSamples=2000]");
            System.err.println("  <dataset-root> must contain train/, val/, test/ subdirectories");
            System.err.println("  [max-samples-per-class] (optional) limit images per class (e.g. 2000, or 0 for all)");
            System.exit(1);
        }
        int maxSamples = 0;
        if (args.length > 1) {
            try {
                maxSamples = Integer.parseInt(args[1]);
            } catch (NumberFormatException ignored) {}
        }
        if (maxSamples <= 0) {
            String prop = System.getProperty("maxSamples");
            if (prop != null && !prop.isEmpty()) {
                try {
                    maxSamples = Integer.parseInt(prop);
                } catch (NumberFormatException ignored) {}
            }
        }

        System.out.println("=== DriveGuard AI — Eye SVM Trainer ===");
        System.out.println("Dataset root: " + dataRoot);
        if (maxSamples > 0) {
            System.out.println("Max samples per class: " + maxSamples);
        } else {
            System.out.println("Samples per class: ALL available");
        }
        System.out.println();

        // Validate dataset root
        validateDir(dataRoot + "/train/awake");
        validateDir(dataRoot + "/train/sleepy");
        validateDir(dataRoot + "/val/awake");
        validateDir(dataRoot + "/val/sleepy");
        validateDir(dataRoot + "/test/awake");
        validateDir(dataRoot + "/test/sleepy");

        EyeFeatureExtractor extractor = new EyeFeatureExtractor(32, 32, true);
        System.out.println("Feature Extractor: Square 32x32 (1:1 aspect ratio) + Histogram Equalization");
        System.out.println("HOG feature vector length: " + extractor.getFeatureLength());
        System.out.println();

        // ── Load training data ────────────────────────────────────────────
        System.out.println("Loading training data...");
        LoadedData trainData = loadDataset(dataRoot + "/train", extractor, "train", maxSamples);
        System.out.println("  [awake]  " + trainData.awakeCount  + " images loaded");
        System.out.println("  [sleepy] " + trainData.sleepyCount + " images loaded");
        System.out.println("Training samples: " + trainData.totalCount()
                + ", feature length: " + extractor.getFeatureLength());
        System.out.println();

        if (trainData.totalCount() == 0) {
            System.err.println("[ERROR] No training images were loaded. Check dataset path.");
            System.exit(1);
        }

        // ── Train SVM ─────────────────────────────────────────────────────
        System.out.println("Training SVM (LINEAR kernel, trainAuto with 5-fold CV)...");
        System.out.flush();
        SVM svm = trainSvm(trainData);
        System.out.println("SVM trained successfully.");

        // ── Save model ────────────────────────────────────────────────────
        new File(MODEL_DIR).mkdirs();
        svm.save(MODEL_FILE);
        System.out.println("Model saved: " + MODEL_FILE);
        saveConfig(CONFIG_FILE, extractor);
        System.out.println("Config saved: " + CONFIG_FILE);
        System.out.println();

        // Release training data mats to free memory before evaluation
        trainData.features.release();
        trainData.labels.release();

        // ── Validation ────────────────────────────────────────────────────
        System.out.println("=== VALIDATION ===");
        LoadedData valData = loadDataset(dataRoot + "/val", extractor, "val", 0);
        evaluate(svm, valData, "Validation");
        valData.features.release();
        valData.labels.release();
        System.out.println();

        // ── Final Test ────────────────────────────────────────────────────
        System.out.println("=== FINAL TEST RESULTS ===");
        LoadedData testData = loadDataset(dataRoot + "/test", extractor, "test", 0);
        evaluate(svm, testData, "Test");
        testData.features.release();
        testData.labels.release();

        System.out.println();
        System.out.println("Done. All results above are from real inference on held-out data.");
    }

    // ── Training ──────────────────────────────────────────────────────────

    /**
     * Configures and trains the SVM using LINEAR kernel and 5-fold CV to select C.
     */
    private static SVM trainSvm(LoadedData data) {
        SVM svm = SVM.create();
        svm.setType(SVM.C_SVC);
        svm.setKernel(SVM.LINEAR);
        svm.setTermCriteria(new TermCriteria(
                TermCriteria.MAX_ITER + TermCriteria.EPS, 1000, 1e-6));

        System.out.println("  Running trainAuto() (5-fold CV to select optimal C for LINEAR SVM)...");
        System.out.flush();

        ParamGrid cgrid = SVM.getDefaultGridPtr(SVM.C);
        ParamGrid ggrid = ParamGrid.create(1, 1, 1);

        boolean ok = false;
        try {
            ok = svm.trainAuto(
                    data.features,
                    Ml.ROW_SAMPLE,
                    data.labels,
                    5,          // 5-fold CV
                    cgrid,
                    ggrid
            );
        } catch (Exception e) {
            System.err.println("  [trainAuto exception] " + e.getMessage());
        }

        if (!ok) {
            System.err.println("[WARN] trainAuto() failed — falling back to standard train() with C=1.0");
            svm.setC(1.0);
            svm.train(data.features, Ml.ROW_SAMPLE, data.labels);
        }

        System.out.printf("  SVM parameters: C=%.4f (Kernel: LINEAR)%n", svm.getC());
        return svm;
    }

    // ── Evaluation ────────────────────────────────────────────────────────

    /**
     * Run inference on a held-out set and print a full metrics report.
     */
    private static void evaluate(SVM svm, LoadedData data, String setName) {
        if (data.totalCount() == 0) {
            System.out.println("  [SKIP] No " + setName + " samples found.");
            return;
        }

        System.out.println("Evaluating on " + setName.toLowerCase()
                + " set (" + data.totalCount() + " samples)...");

        long tp = 0, tn = 0, fp = 0, fn = 0;

        for (int i = 0; i < data.features.rows(); i++) {
            Mat sample = data.features.row(i);
            int predicted = (int) svm.predict(sample);
            int actual    = (int) data.labels.get(i, 0)[0];

            if (predicted == LABEL_SLEEPY && actual == LABEL_SLEEPY) tp++;
            else if (predicted == LABEL_AWAKE  && actual == LABEL_AWAKE)  tn++;
            else if (predicted == LABEL_SLEEPY && actual == LABEL_AWAKE)  fp++;
            else if (predicted == LABEL_AWAKE  && actual == LABEL_SLEEPY) fn++;
        }

        long total    = tp + tn + fp + fn;
        double acc    = 100.0 * (tp + tn) / total;
        double prec   = (tp + fp) > 0 ? 100.0 * tp / (tp + fp) : 0;
        double rec    = (tp + fn) > 0 ? 100.0 * tp / (tp + fn) : 0;
        double f1     = (prec + rec) > 0 ? 2.0 * prec * rec / (prec + rec) : 0;

        double awakePrec = (tn + fn) > 0 ? 100.0 * tn / (tn + fn) : 0;
        double awakeRec  = (tn + fp) > 0 ? 100.0 * tn / (tn + fp) : 0;

        System.out.printf("Overall Accuracy:   %.2f%%%n",  acc);
        System.out.printf("Class SLEEPY (Pos): Precision = %.2f%%, Recall = %.2f%%, F1 = %.2f%%%n", prec, rec, f1);
        System.out.printf("Class AWAKE  (Neg): Precision = %.2f%%, Recall = %.2f%%%n", awakePrec, awakeRec);
        System.out.println();
        System.out.println("Confusion Matrix:");
        System.out.printf("                Pred AWAKE   Pred SLEEPY%n");
        System.out.printf("Actual AWAKE    %-12d %-12d (Recall: %.2f%%)%n", tn, fp, awakeRec);
        System.out.printf("Actual SLEEPY   %-12d %-12d (Recall: %.2f%%)%n", fn, tp, rec);
    }

    // ── Dataset Loading ───────────────────────────────────────────────────

    /** Holds the stacked feature Mat and corresponding label Mat. */
    private static class LoadedData {
        Mat features;   // N × featureLen, CV_32F
        Mat labels;     // N × 1, CV_32S
        int awakeCount  = 0;
        int sleepyCount = 0;
        int totalCount() { return awakeCount + sleepyCount; }
    }

    /**
     * Loads images from splitDir/awake and splitDir/sleepy.
     * Images are extracted into rows of a preallocated feature Mat.
     *
     * @param splitDir path to the split root (e.g. ".../data/train")
     * @param extractor shared extractor (caches feature length)
     * @param splitName label for progress messages
     * @param maxSamples max images per class (0 for all)
     */
    private static LoadedData loadDataset(String splitDir,
                                          EyeFeatureExtractor extractor,
                                          String splitName,
                                          int maxSamples) throws IOException {
        LoadedData result = new LoadedData();

        List<float[]> featureRows = new ArrayList<>();
        List<Integer> labelList   = new ArrayList<>();

        // Load awake images
        int awake  = loadFolder(splitDir + "/awake",  LABEL_AWAKE,  extractor,
                                featureRows, labelList, splitName + "/awake", maxSamples);
        // Load sleepy images
        int sleepy = loadFolder(splitDir + "/sleepy", LABEL_SLEEPY, extractor,
                                featureRows, labelList, splitName + "/sleepy", maxSamples);

        result.awakeCount  = awake;
        result.sleepyCount = sleepy;

        if (featureRows.isEmpty()) {
            result.features = new Mat();
            result.labels   = new Mat();
            return result;
        }

        int flen = featureRows.get(0).length;
        int n    = featureRows.size();

        // Stack feature rows into a single N × flen Mat (CV_32F)
        result.features = new Mat(n, flen, CvType.CV_32F);
        result.labels   = new Mat(n, 1,    CvType.CV_32S);

        for (int i = 0; i < n; i++) {
            result.features.put(i, 0, featureRows.get(i));
            result.labels.put(i, 0, new int[]{ labelList.get(i) });
        }

        return result;
    }

    /**
     * Walk a single folder, extract HOG features, append to the lists.
     *
     * @return the number of images successfully loaded
     */
    private static int loadFolder(String folderPath,
                                   int label,
                                   EyeFeatureExtractor extractor,
                                   List<float[]> featureRows,
                                   List<Integer> labelList,
                                   String progressTag,
                                   int maxSamples) throws IOException {

        File folder = new File(folderPath);
        if (!folder.exists() || !folder.isDirectory()) {
            System.err.println("[WARN] Folder not found, skipping: " + folderPath);
            return 0;
        }

        List<Path> imagePaths;
        try (Stream<Path> walk = Files.walk(Paths.get(folderPath))) {
            imagePaths = walk
                    .filter(Files::isRegularFile)
                    .filter(EyeSvmTrainer::isImageFile)
                    .sorted()   // deterministic ordering across runs
                    .toList();
        }

        int count   = 0;
        int skipped = 0;
        int total   = imagePaths.size();
        if (maxSamples > 0 && total > maxSamples) {
            total = maxSamples;
        }
        int reportEvery = Math.max(1, total / 10);  // report ~10 times

        System.out.println("  Loading " + progressTag + " (" + total + " files)...");

        for (Path p : imagePaths) {
            if (maxSamples > 0 && count >= maxSamples) {
                break;
            }

            Mat row = extractor.extract(p.toString());
            if (row == null) {
                skipped++;
                continue;
            }

            // Convert Mat row → float[]
            float[] rowData = new float[(int) row.total()];
            row.get(0, 0, rowData);
            featureRows.add(rowData);
            labelList.add(label);
            row.release();
            count++;

            if (count % reportEvery == 0) {
                System.out.printf("    %d / %d%n", count, total);
                System.out.flush();
            }
        }

        if (skipped > 0) {
            System.out.println("  [WARN] Skipped " + skipped + " unreadable images in " + progressTag);
        }

        return count;
    }

    // ── Utilities ─────────────────────────────────────────────────────────

    private static boolean isImageFile(Path p) {
        String name = p.getFileName().toString().toLowerCase();
        for (String ext : SUPPORTED_EXTS) {
            if (name.endsWith(ext)) return true;
        }
        return false;
    }

    private static void validateDir(String path) {
        File f = new File(path);
        if (!f.exists() || !f.isDirectory()) {
            System.err.println("[ERROR] Required directory not found: " + path);
            System.err.println("        Ensure dataset root contains train/awake, train/sleepy, etc.");
            System.exit(1);
        }
    }

    private static void saveConfig(String configPath, EyeFeatureExtractor extractor) throws IOException {
        new File(MODEL_DIR).mkdirs();
        try (PrintWriter pw = new PrintWriter(new FileWriter(configPath))) {
            for (String line : extractor.getConfigLines()) {
                pw.println(line);
            }
            pw.println("hog.feature.length=" + extractor.getFeatureLength());
        }
    }
}
