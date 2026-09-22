package com.driveguard;

import com.driveguard.detection.DrowsinessDetector;
import com.driveguard.model.DetectionResult;
import com.driveguard.ml.EyeSvmClassifier.Prediction;

public class TestDualEyeLogic {
    public static void main(String[] args) {
        System.out.println("--- Testing Dual-Eye Logic and State Machine ---");
        DrowsinessDetector detector = new DrowsinessDetector();

        // 1. Both eyes AWAKE -> counter stays 0, status ACTIVE
        for (int i = 0; i < 5; i++) {
            DetectionResult r = detector.process(true, 2, Prediction.AWAKE, Prediction.AWAKE, Prediction.AWAKE);
            assert r.getClosedFrameCount() == 0 : "Count should be 0";
            assert r.getDriverStatus() == DetectionResult.DriverStatus.ACTIVE : "Status should be ACTIVE";
        }
        System.out.println("Test 1 Passed: Both eyes AWAKE -> ACTIVE, count = 0");

        // 2. One eye AWAKE, one eye SLEEPY -> counter does not increment towards DROWSY
        for (int i = 0; i < 25; i++) {
            DetectionResult r = detector.process(true, 1, Prediction.AWAKE, Prediction.AWAKE, Prediction.SLEEPY);
            assert r.getClosedFrameCount() == 0 : "Count should remain 0 for mixed eyes";
            assert r.getDriverStatus() == DetectionResult.DriverStatus.ACTIVE : "Status should stay ACTIVE";
        }
        System.out.println("Test 2 Passed: Mixed eyes (one awake, one sleepy) -> ACTIVE, count = 0 (No false alert)");

        // 3. Haar detects 0 eyes, but SVM sees both AWAKE -> should NOT trigger DROWSY!
        for (int i = 0; i < 25; i++) {
            DetectionResult r = detector.process(true, 0, Prediction.AWAKE, Prediction.AWAKE, Prediction.AWAKE);
            assert r.getClosedFrameCount() == 0 : "Haar 0 eyes must not increment counter when SVM is AWAKE";
            assert r.getDriverStatus() == DetectionResult.DriverStatus.ACTIVE : "Status should stay ACTIVE";
        }
        System.out.println("Test 3 Passed: Haar 0-eye detection does NOT trigger false alert when eyes are open");

        // 4. Both eyes SLEEPY (intentional closure) -> increments to 20 -> DROWSY
        for (int i = 1; i <= 20; i++) {
            DetectionResult r = detector.process(true, 0, Prediction.SLEEPY, Prediction.SLEEPY, Prediction.SLEEPY);
            assert r.getClosedFrameCount() == i : "Count should be " + i;
            if (i < 20) {
                assert r.getDriverStatus() == DetectionResult.DriverStatus.ACTIVE : "Status before 20 should be ACTIVE";
            } else {
                assert r.getDriverStatus() == DetectionResult.DriverStatus.DROWSY : "Status at 20 should be DROWSY";
            }
        }
        System.out.println("Test 4 Passed: Both eyes SLEEPY -> smoothly triggers DROWSY at frame 20");

        // 5. Driver opens eyes -> immediate reset to ACTIVE, count = 0
        DetectionResult rRecovery = detector.process(true, 2, Prediction.AWAKE, Prediction.AWAKE, Prediction.AWAKE);
        assert rRecovery.getClosedFrameCount() == 0 : "Count should reset to 0 immediately";
        assert rRecovery.getDriverStatus() == DetectionResult.DriverStatus.ACTIVE : "Status should return to ACTIVE";
        System.out.println("Test 5 Passed: Recovery on open eyes -> ACTIVE, count = 0");

        System.out.println("\nALL STATE MACHINE UNIT TESTS PASSED SUCCESSFULLY!");
    }
}
