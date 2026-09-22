package com.driveguard.ml;

import nu.pattern.OpenCV;
import org.opencv.core.*;
import org.opencv.ml.Ml;
import org.opencv.ml.ParamGrid;
import org.opencv.ml.SVM;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

/**
 * Controlled investigation script comparing HOG configurations and SVM kernels
 * on the exact same dataset splits.
 */
public class ModelInvestigation {

    private static final int LABEL_AWAKE  = 0;
    private static final int LABEL_SLEEPY = 1;
    private static final String[] SUPPORTED_EXTS = { ".png", ".jpg", ".jpeg", ".bmp" };

    public static class Metrics {
        String name;
        long tp, tn, fp, fn;
        double accuracy, precision, recall, f1;
        double awakePrecision, awakeRecall;
        double C, gamma;

        public void compute(String name, long tp, long tn, long fp, long fn, double C, double gamma) {
            this.name = name;
            this.tp = tp;
            this.tn = tn;
            this.fp = fp;
            this.fn = fn;
            this.C = C;
            this.gamma = gamma;

            long total = tp + tn + fp + fn;
            this.accuracy = total > 0 ? 100.0 * (tp + tn) / total : 0;
            this.precision = (tp + fp) > 0 ? 100.0 * tp / (tp + fp) : 0;
            this.recall = (tp + fn) > 0 ? 100.0 * tp / (tp + fn) : 0;
            this.f1 = (precision + recall) > 0 ? 2.0 * precision * recall / (precision + recall) : 0;

            this.awakePrecision = (tn + fn) > 0 ? 100.0 * tn / (tn + fn) : 0;
            this.awakeRecall = (tn + fp) > 0 ? 100.0 * tn / (tn + fp) : 0;
        }

        public void printReport() {
            System.out.println("--------------------------------------------------------------------------------");
            System.out.println("CONFIGURATION: " + name);
            System.out.printf("Parameters: C = %.4f, Gamma = %.4f%n", C, gamma);
            System.out.printf("Accuracy:   %.2f%%%n", accuracy);
            System.out.printf("Precision (Sleepy): %.2f%%%n", precision);
            System.out.printf("Recall    (Sleepy): %.2f%%%n", recall);
            System.out.printf("F1-Score  (Sleepy): %.2f%%%n", f1);
            System.out.printf("Awake-Precision:    %.2f%%%n", awakePrecision);
            System.out.printf("Awake-Recall:       %.2f%%%n", awakeRecall);
            System.out.println("Confusion Matrix (Positive = SLEEPY):");
            System.out.printf("                Pred AWAKE   Pred SLEEPY%n");
            System.out.printf("Actual AWAKE    %-12d %-12d (Recall: %.2f%%)%n", tn, fp, awakeRecall);
            System.out.printf("Actual SLEEPY   %-12d %-12d (Recall: %.2f%%)%n", fn, tp, recall);
            System.out.println("--------------------------------------------------------------------------------");
        }
    }

    public static void main(String[] args) throws IOException {
        OpenCV.loadLocally();

        String dataRoot = args.length > 0 ? args[0] : System.getProperty("datasetPath");
        if (dataRoot == null) {
            dataRoot = "C:\\Users\\Merin Joys\\Merin\\merin\\Datasetsdownloads\\data";
        }

        int sampleLimit = 1000;
        if (args.length > 1) {
            try { sampleLimit = Integer.parseInt(args[1]); } catch (Exception ignored) {}
        }

        System.out.println("================================================================================");
        System.out.println("        DRIVEGUARD AI — MODEL IMPROVEMENT CONTROLLED INVESTIGATION              ");
        System.out.println("================================================================================");
        System.out.println("Dataset root: " + dataRoot);
        System.out.println("Samples per class: " + sampleLimit + " train, " + sampleLimit + " validation");
        System.out.println();

        // 1. Collect file paths so ALL experiments use the exact identical image files
        List<String> trainAwakePaths  = getImageFiles(dataRoot + "/train/awake", sampleLimit);
        List<String> trainSleepyPaths = getImageFiles(dataRoot + "/train/sleepy", sampleLimit);
        List<String> valAwakePaths    = getImageFiles(dataRoot + "/val/awake", sampleLimit);
        List<String> valSleepyPaths   = getImageFiles(dataRoot + "/val/sleepy", sampleLimit);

        System.out.printf("Collected %d train awake, %d train sleepy images.%n", trainAwakePaths.size(), trainSleepyPaths.size());
        System.out.printf("Collected %d val awake, %d val sleepy images.%n", valAwakePaths.size(), valSleepyPaths.size());
        System.out.println();

        List<Metrics> allResults = new ArrayList<>();

        // Experiment A: Current HOG (64x32) + RBF SVM
        allResults.add(runExperiment(
                "A: HOG 64x32 (Squashed) + RBF SVM",
                new EyeFeatureExtractor(64, 32, false),
                SVM.RBF,
                trainAwakePaths, trainSleepyPaths, valAwakePaths, valSleepyPaths
        ));

        // Experiment B: Current HOG (64x32) + LINEAR SVM
        allResults.add(runExperiment(
                "B: HOG 64x32 (Squashed) + LINEAR SVM",
                new EyeFeatureExtractor(64, 32, false),
                SVM.LINEAR,
                trainAwakePaths, trainSleepyPaths, valAwakePaths, valSleepyPaths
        ));

        // Experiment C1: Square HOG 32x32 + LINEAR SVM
        allResults.add(runExperiment(
                "C1: Square HOG 32x32 (1:1 Ratio) + LINEAR SVM",
                new EyeFeatureExtractor(32, 32, false),
                SVM.LINEAR,
                trainAwakePaths, trainSleepyPaths, valAwakePaths, valSleepyPaths
        ));

        // Experiment C2: Square HOG 32x32 + RBF SVM
        allResults.add(runExperiment(
                "C2: Square HOG 32x32 (1:1 Ratio) + RBF SVM",
                new EyeFeatureExtractor(32, 32, false),
                SVM.RBF,
                trainAwakePaths, trainSleepyPaths, valAwakePaths, valSleepyPaths
        ));

        // Experiment C3: Square HOG 48x48 + LINEAR SVM
        allResults.add(runExperiment(
                "C3: Square HOG 48x48 (1:1 Ratio) + LINEAR SVM",
                new EyeFeatureExtractor(48, 48, false),
                SVM.LINEAR,
                trainAwakePaths, trainSleepyPaths, valAwakePaths, valSleepyPaths
        ));

        // Experiment C4: Square HOG 48x48 + RBF SVM
        allResults.add(runExperiment(
                "C4: Square HOG 48x48 (1:1 Ratio) + RBF SVM",
                new EyeFeatureExtractor(48, 48, false),
                SVM.RBF,
                trainAwakePaths, trainSleepyPaths, valAwakePaths, valSleepyPaths
        ));

        // Experiment C5: Square HOG 32x32 + HistEqualization + LINEAR SVM
        allResults.add(runExperiment(
                "C5: Square HOG 32x32 + EqualizeHist + LINEAR SVM",
                new EyeFeatureExtractor(32, 32, true),
                SVM.LINEAR,
                trainAwakePaths, trainSleepyPaths, valAwakePaths, valSleepyPaths
        ));

        // Summary Table
        System.out.println();
        System.out.println("================================================================================================================");
        System.out.println("                                      CONTROLLED EXPERIMENT COMPARISON SUMMARY                                  ");
        System.out.println("================================================================================================================");
        System.out.printf("%-45s | %-9s | %-9s | %-9s | %-9s | %-12s | %-12s%n",
                "Experiment Configuration", "Accuracy", "Precision", "Recall", "F1-Score", "Awake-Recall", "Sleepy-Rec");
        System.out.println("----------------------------------------------------------------------------------------------------------------");
        for (Metrics m : allResults) {
            System.out.printf("%-45s | %8.2f%% | %8.2f%% | %8.2f%% | %8.2f%% | %11.2f%% | %11.2f%%%n",
                    m.name, m.accuracy, m.precision, m.recall, m.f1, m.awakeRecall, m.recall);
        }
        System.out.println("================================================================================================================");
    }

    private static Metrics runExperiment(String name,
                                         EyeFeatureExtractor extractor,
                                         int kernelType,
                                         List<String> trainAwake,
                                         List<String> trainSleepy,
                                         List<String> valAwake,
                                         List<String> valSleepy) {
        System.out.println("\n>>> Running " + name + " (feature length: " + extractor.getFeatureLength() + ")...");
        long t0 = System.currentTimeMillis();

        // 1. Extract Training Features
        int totalTrain = trainAwake.size() + trainSleepy.size();
        Mat trainFeatures = new Mat(totalTrain, extractor.getFeatureLength(), CvType.CV_32F);
        Mat trainLabels   = new Mat(totalTrain, 1, CvType.CV_32S);

        int row = 0;
        for (String p : trainAwake) {
            Mat feat = extractor.extract(p);
            if (feat != null) {
                float[] buf = new float[(int) feat.total()];
                feat.get(0, 0, buf);
                trainFeatures.put(row, 0, buf);
                trainLabels.put(row, 0, new int[]{ LABEL_AWAKE });
                feat.release();
                row++;
            }
        }
        for (String p : trainSleepy) {
            Mat feat = extractor.extract(p);
            if (feat != null) {
                float[] buf = new float[(int) feat.total()];
                feat.get(0, 0, buf);
                trainFeatures.put(row, 0, buf);
                trainLabels.put(row, 0, new int[]{ LABEL_SLEEPY });
                feat.release();
                row++;
            }
        }

        // 2. Train SVM
        SVM svm = SVM.create();
        svm.setType(SVM.C_SVC);
        svm.setKernel(kernelType);
        svm.setTermCriteria(new TermCriteria(TermCriteria.MAX_ITER + TermCriteria.EPS, 1000, 1e-6));

        ParamGrid cgrid = SVM.getDefaultGridPtr(SVM.C);
        ParamGrid ggrid = (kernelType == SVM.RBF) ? SVM.getDefaultGridPtr(SVM.GAMMA) : ParamGrid.create(1, 1, 1);

        try {
            svm.trainAuto(trainFeatures, Ml.ROW_SAMPLE, trainLabels, 5, cgrid, ggrid);
        } catch (Exception e) {
            System.err.println("trainAuto exception: " + e.getMessage() + ", falling back to train()");
            svm.setC(1.0);
            svm.setGamma(0.5);
            svm.train(trainFeatures, Ml.ROW_SAMPLE, trainLabels);
        }

        trainFeatures.release();
        trainLabels.release();

        // 3. Evaluate on Validation Set
        long tp = 0, tn = 0, fp = 0, fn = 0;

        for (String p : valAwake) {
            Mat feat = extractor.extract(p);
            if (feat != null) {
                int pred = (int) svm.predict(feat);
                if (pred == LABEL_AWAKE) tn++;
                else fp++;
                feat.release();
            }
        }

        for (String p : valSleepy) {
            Mat feat = extractor.extract(p);
            if (feat != null) {
                int pred = (int) svm.predict(feat);
                if (pred == LABEL_SLEEPY) tp++;
                else fn++;
                feat.release();
            }
        }

        Metrics m = new Metrics();
        m.compute(name, tp, tn, fp, fn, svm.getC(), svm.getGamma());
        m.printReport();
        System.out.printf("Completed in %.1f seconds.%n", (System.currentTimeMillis() - t0) / 1000.0);
        return m;
    }

    private static List<String> getImageFiles(String dirPath, int maxFiles) throws IOException {
        List<String> list = new ArrayList<>();
        File dir = new File(dirPath);
        if (!dir.exists() || !dir.isDirectory()) return list;

        try (Stream<Path> s = Files.walk(Paths.get(dirPath))) {
            s.filter(Files::isRegularFile)
             .map(Path::toString)
             .filter(ModelInvestigation::isImage)
             .sorted()
             .limit(maxFiles)
             .forEach(list::add);
        }
        return list;
    }

    private static boolean isImage(String s) {
        String lower = s.toLowerCase();
        for (String ext : SUPPORTED_EXTS) {
            if (lower.endsWith(ext)) return true;
        }
        return false;
    }
}
