package com.driveguard;

import nu.pattern.OpenCV;

/**
 * Non-JavaFX entry point.
 *
 * IMPORTANT: OpenCV native library MUST be loaded before any JavaFX class
 * is referenced. This separate Main class ensures that order.
 *
 * Run with: mvn javafx:run
 */
public class Main {

    public static void main(String[] args) {
        // Load the OpenCV native library (Windows x64 DLL is bundled in the
        // openpnp artifact and extracted automatically to a temp directory).
        OpenCV.loadShared();

        // Now launch the JavaFX application
        MainApp.launch(MainApp.class, args);
    }
}
