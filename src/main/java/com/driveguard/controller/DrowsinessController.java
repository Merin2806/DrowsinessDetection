package com.driveguard.controller;

import com.driveguard.alert.AlertService;
import com.driveguard.camera.CameraService;
import com.driveguard.detection.DrowsinessDetector;
import com.driveguard.detection.EyeDetector;
import com.driveguard.detection.FaceDetector;
import com.driveguard.model.DetectionResult;
import com.driveguard.model.DetectionResult.DriverStatus;

import javafx.application.Platform;
import javafx.fxml.FXML;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.layout.VBox;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;

import java.io.IOException;

/**
 * JavaFX FXML controller for the DriveGuard AI dashboard.
 *
 * Responsibilities:
 * ─ Wire button actions (Start / Stop / Exit)
 * ─ Receive per-frame callbacks from CameraService (on FX thread)
 * ─ Update status labels and CSS style classes
 * ─ Manage alert transitions (start alert only on ACTIVE→DROWSY transition)
 * ─ Clean shutdown on application close
 */
public class DrowsinessController {

    // ── FXML-injected nodes ───────────────────────────────────────────────────
    @FXML private ImageView cameraView;
    @FXML private VBox      placeholderLabel;  // VBox in FXML, contains icon + text labels

    @FXML private Label driverStatusValueLabel;
    @FXML private Label eyeStatusValueLabel;
    @FXML private Label faceDetectionValueLabel;
    @FXML private Label monitoringValueLabel;
    @FXML private Label statusBarLabel;

    @FXML private Button startButton;
    @FXML private Button stopButton;

    // ── Services ──────────────────────────────────────────────────────────────
    private CameraService   cameraService;
    private AlertService    alertService;
    private FaceDetector    faceDetector;
    private EyeDetector     eyeDetector;

    /** Track the previous driver status so we only trigger/stop alert on transitions. */
    private DriverStatus previousStatus = DriverStatus.WAITING;

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    @FXML
    public void initialize() {
        alertService = new AlertService();
        applyStoppedState();

        // Pre-load cascade classifiers in a background thread.
        // This avoids a brief UI freeze when the user first clicks START.
        Thread initThread = new Thread(() -> {
            try {
                faceDetector = new FaceDetector();
                eyeDetector  = new EyeDetector();
                Platform.runLater(() ->
                    statusBarLabel.setText("Ready. Press START CAMERA to begin monitoring."));
            } catch (IOException e) {
                Platform.runLater(() -> showError(
                    "Cascade Load Error",
                    "Could not load Haar cascade XML files.\n\n"
                    + e.getMessage()));
            }
        }, "DriveGuard-InitThread");
        initThread.setDaemon(true);
        initThread.start();
    }

    // ── Button Handlers ───────────────────────────────────────────────────────

    @FXML
    private void onStartCamera() {
        if (faceDetector == null || eyeDetector == null) {
            showError("Not Ready",
                "Detectors are still initializing — please wait a moment and try again.");
            return;
        }

        try {
            DrowsinessDetector drowsinessDetector = new DrowsinessDetector();
            cameraService = new CameraService(
                    faceDetector, eyeDetector, drowsinessDetector,
                    this::onFrameReceived);
            cameraService.start();

            startButton.setDisable(true);
            stopButton.setDisable(false);
            setMonitoringRunning();

        } catch (IllegalStateException ex) {
            showError("Camera Error", ex.getMessage());
        }
    }

    @FXML
    private void onStopCamera() {
        stopCameraInternal();
        applyStoppedState();
    }

    @FXML
    private void onExit() {
        shutdown();
        Platform.exit();
        System.exit(0);
    }

    // ── Frame Callback (always called on the JavaFX Application Thread) ───────

    private void onFrameReceived(Image image, DetectionResult result) {
        // ── Camera feed ────────────────────────────────────────────────────
        placeholderLabel.setVisible(false);
        cameraView.setImage(image);

        // ── Face Detection card ────────────────────────────────────────────
        if (result.isFaceDetected()) {
            setLabelState(faceDetectionValueLabel, "DETECTED", "status-active");
        } else {
            setLabelState(faceDetectionValueLabel, "NOT DETECTED", "status-warning");
        }

        // ── Eye Status card ────────────────────────────────────────────────
        switch (result.getEyeStatus()) {
            case OPEN        -> setLabelState(eyeStatusValueLabel, "OPEN",         "status-active");
            case CLOSED      -> setLabelState(eyeStatusValueLabel, "CLOSED",       "status-warning");
            case NOT_DETECTED-> setLabelState(eyeStatusValueLabel, "NOT DETECTED", "status-neutral");
        }

        // ── Driver Status card + status bar ───────────────────────────────
        DriverStatus current = result.getDriverStatus();
        switch (current) {
            case ACTIVE -> {
                setLabelState(driverStatusValueLabel, "ACTIVE", "status-active", "large-status");
                setStatusBar("System is monitoring driver alertness...", false);
            }
            case DROWSY -> {
                setLabelState(driverStatusValueLabel, "DROWSY", "status-danger", "large-status");
                setStatusBar("⚠   DROWSINESS ALERT — Prolonged eye closure detected", true);
            }
            case WAITING -> {
                setLabelState(driverStatusValueLabel, "WAITING", "status-neutral", "large-status");
                setStatusBar("Face not detected — please position yourself in front of the camera.", false);
            }
        }

        // ── Alert transition logic ─────────────────────────────────────────
        // Only call start/stop on state *transitions* — not every frame.
        if (current == DriverStatus.DROWSY && previousStatus != DriverStatus.DROWSY) {
            alertService.startAlert();
        } else if (current != DriverStatus.DROWSY && previousStatus == DriverStatus.DROWSY) {
            alertService.stopAlert();
        }
        previousStatus = current;
    }

    // ── UI State Helpers ──────────────────────────────────────────────────────

    private void applyStoppedState() {
        startButton.setDisable(false);
        stopButton.setDisable(true);
        placeholderLabel.setVisible(true);
        cameraView.setImage(null);

        setMonitoringState("STOPPED", "status-neutral");
        setLabelState(driverStatusValueLabel, "—", "status-neutral", "large-status");
        setLabelState(eyeStatusValueLabel,    "—", "status-neutral");
        setLabelState(faceDetectionValueLabel,"—", "status-neutral");
        setStatusBar("Camera stopped. Press START CAMERA to begin monitoring.", false);
        previousStatus = DriverStatus.WAITING;
    }

    private void setMonitoringRunning() {
        setMonitoringState("RUNNING", "status-active");
        setStatusBar("System is monitoring driver alertness...", false);
    }

    private void setMonitoringState(String text, String styleToken) {
        monitoringValueLabel.setText(text);
        monitoringValueLabel.getStyleClass().setAll("status-value", styleToken);
    }

    /**
     * Update a status label's text and CSS style tokens.
     * Always keeps the base "status-value" class plus any extras provided.
     */
    private void setLabelState(Label label, String text, String... extraStyleTokens) {
        label.setText(text);
        label.getStyleClass().setAll("status-value");
        for (String token : extraStyleTokens) {
            label.getStyleClass().add(token);
        }
    }

    private void setStatusBar(String message, boolean isAlert) {
        statusBarLabel.setText(message);
        if (isAlert) {
            statusBarLabel.getStyleClass().setAll("status-bar-label", "alert-bar");
        } else {
            statusBarLabel.getStyleClass().setAll("status-bar-label");
        }
    }

    private void showError(String title, String message) {
        Alert dialog = new Alert(Alert.AlertType.ERROR);
        dialog.setTitle(title);
        dialog.setHeaderText(title);
        dialog.setContentText(message);
        dialog.showAndWait();
    }

    // ── Shutdown ──────────────────────────────────────────────────────────────

    private void stopCameraInternal() {
        if (cameraService != null && cameraService.isRunning()) {
            cameraService.stop();
        }
        if (alertService != null) {
            alertService.stopAlert();
        }
    }

    /** Called by MainApp on window close — ensures clean resource release. */
    public void shutdown() {
        stopCameraInternal();
        if (alertService != null) {
            alertService.shutdown();
        }
    }
}
