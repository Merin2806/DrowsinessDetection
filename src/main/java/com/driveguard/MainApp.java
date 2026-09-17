package com.driveguard;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.stage.Stage;
import com.driveguard.controller.DrowsinessController;

/**
 * JavaFX Application class for DriveGuard AI.
 *
 * Kept separate from Main.java so that the OpenCV native library
 * (loaded in Main.main()) is initialised before any JavaFX classes
 * are touched — this order is required on some JVM configurations.
 */
public class MainApp extends Application {

    @Override
    public void start(Stage primaryStage) throws Exception {
        // Load FXML (path is relative to this class's package in the classpath)
        FXMLLoader loader = new FXMLLoader(
                getClass().getResource("main.fxml"));
        Parent root = loader.load();

        // Build scene and attach the dark stylesheet
        Scene scene = new Scene(root, 1200, 720);
        scene.getStylesheets().add(
                getClass().getResource("/styles/dark-theme.css").toExternalForm());

        // Window setup
        primaryStage.setTitle("DriveGuard AI  —  Real-Time Driver Drowsiness Detection");
        primaryStage.setScene(scene);
        primaryStage.setMinWidth(1000);
        primaryStage.setMinHeight(680);

        // Register shutdown hook so the camera and audio are released cleanly
        // when the user closes the window via the X button.
        DrowsinessController controller = loader.getController();
        primaryStage.setOnCloseRequest(event -> controller.shutdown());

        primaryStage.show();
    }

    // Called by Main.main() — do NOT rename or remove.
    public static void main(String[] args) {
        launch(args);
    }
}
