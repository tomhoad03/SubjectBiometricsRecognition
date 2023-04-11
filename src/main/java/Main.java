import ai.djl.Application;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.Joints;
import ai.djl.repository.zoo.Criteria;
import ai.djl.translate.TranslateException;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.feature.DoubleFVComparison;
import org.openimaj.feature.FeatureVector;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.colour.RGBColour;
import org.openimaj.ml.pca.FeatureVectorPCA;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.concurrent.atomic.AtomicReference;

public class Main {
    private static final String PATH = Paths.get("").toAbsolutePath() + "\\src\\main\\java\\";
    private static Predictor<Image, Joints> predictor;
    private static final Float[][] temperatures = new Float[48][];

    public static void main(String[] args) throws IOException, TranslateException {
        AtomicReference<VFSListDataset<MBFImage>> training = new AtomicReference<>(new VFSListDataset<>(PATH + "biometrics\\training", ImageUtilities.MBFIMAGE_READER));
        AtomicReference<VFSListDataset<MBFImage>> testing = new AtomicReference<>(new VFSListDataset<>(PATH + "biometrics\\testing", ImageUtilities.MBFIMAGE_READER));

        ArrayList<ComputedImage> trainingImages = new ArrayList<>();
        ArrayList<ComputedImage> testingImages = new ArrayList<>();
        ArrayList<FeatureVector> featureVectors = new ArrayList<>();

        long startTime = System.currentTimeMillis();

        // Pose estimation using DJL
        try {
            predictor = Criteria.builder()
                    .optApplication(Application.CV.POSE_ESTIMATION)
                    .setTypes(Image.class, Joints.class)
                    .optFilter("backbone", "resnet18")
                    .optFilter("flavor", "v1b")
                    .optFilter("dataset", "imagenet")
                    .optEngine("MXNet")
                    .build()
                    .loadModel()
                    .newPredictor();
        } catch (Exception e) {
            e.printStackTrace();
        }

        // Colour generation
        for (int i = 0; i < temperatures.length; i++) {
            temperatures[i] = RGBColour.randomColour();
        }

        // Read and print the training images
        int id = 1;
        for (MBFImage trainingImage : training.get()) {
            trainingImages.add(new ComputedImage(id, trainingImage, true, PATH, predictor, temperatures));
            id++;
        }

        // Read and print the testing images
        id = 1;
        for (MBFImage testingImage : testing.get()) {
            testingImages.add(new ComputedImage(id, testingImage, false, PATH, predictor, temperatures));
            id++;
        }

        // Learning PCA basis
        for (ComputedImage trainingImage : trainingImages) {
            featureVectors.add(trainingImage.getExtractedFeature());
        }
        FeatureVectorPCA pca = new FeatureVectorPCA();
        pca.learnBasis(featureVectors);

        // Nearest neighbour to find the closest training image to each testing image
        float correctCount = 0f;

        for (ComputedImage testingImage : testingImages) {
            ComputedImage nearestImage = null;
            double nearestDistance = -1, furthestDistance = -1;

            // Finds the nearest image
            for (ComputedImage trainingImage : trainingImages) {
                double distance = DoubleFVComparison.EUCLIDEAN.compare(pca.project(trainingImage.getExtractedFeature()), pca.project(testingImage.getExtractedFeature()));

                if (nearestDistance == -1 || distance < nearestDistance) {
                    nearestDistance = distance;
                    nearestImage = trainingImage;
                }

                if (furthestDistance == -1 || distance > furthestDistance) {
                    furthestDistance = distance;
                }
            }

            // Checks classification accuracy
            if (nearestImage != null && classificationCheck(testingImage.getId(), nearestImage.getId())) {
                System.out.print(testingImage.getId() + " ");
                correctCount += 1f;
            }
        }

        // Histogram of distances
        double correctClassificationRate = (correctCount / 22f) * 100f;
        ArrayList<Double> intraDistances = new ArrayList<>(), interDistances = new ArrayList<>();

        for (int i = 0; i < trainingImages.size(); i++) {
            for (int j = i; j < trainingImages.size(); j++) {
                ComputedImage trainingImageA = trainingImages.get(i);
                ComputedImage trainingImageB = trainingImages.get(j);

                if (trainingImageA.getId() != trainingImageB.getId()) {
                    double distance = DoubleFVComparison.EUCLIDEAN.compare(pca.project(trainingImageA.getExtractedFeature()), pca.project(trainingImageB.getExtractedFeature()));

                    if (verificationCheck(trainingImageA.getId(), trainingImageB.getId())) {
                        interDistances.add(distance);
                    } else {
                        intraDistances.add(distance);
                    }
                }
            }
        }

        interDistances.sort(Comparator.comparingDouble(o -> o));
        intraDistances.sort(Comparator.comparingDouble(o -> o));
        double EER = 0f;
        double smallestDistance = -1f;

        // EER
        for (double threshold = 0f; threshold < 1f; threshold += 0.000001f) {
            double tempThreshold = threshold;
            double FAR = interDistances.stream().filter(a -> a > tempThreshold).count() / (double) interDistances.size();
            double FFR = intraDistances.stream().filter(a -> a < tempThreshold).count() / (double) intraDistances.size();

            if (FAR == FFR || smallestDistance == -1f || Math.abs(FAR - FFR) < smallestDistance) {
                EER = FAR * 100f;
                smallestDistance = Math.abs(FAR - FFR);
            }

            if (FAR == FFR) {
                break;
            }
        }
        long endTime = System.currentTimeMillis();

        // Print the results
        String results = "Correct Classification Rate (CCR) = " + (float) correctClassificationRate + "%"
                + "\n" + "Equal Error Rate: " + (float) EER + "%"
                + "\n" + "Duration: " + (endTime - startTime) + "ms";

        File resultsFile = new File(PATH + "\\results.txt");
        FileWriter fileWriter = new FileWriter(resultsFile);
        fileWriter.write(results);
        fileWriter.close();

        System.out.println("\n" + results);
    }

    // Classification check
    static boolean classificationCheck(int testingId, int trainingId) {
        return switch (testingId) {
            case 1, 2 -> trainingId == 47 || trainingId == 48;
            case 3, 4 -> trainingId == 49 || trainingId == 50;
            case 5, 6 -> trainingId == 51 || trainingId == 52; // y
            case 7, 8 -> trainingId == 53 || trainingId == 54; // y
            case 9, 10 -> trainingId == 55 || trainingId == 56; // y
            case 11, 12 -> trainingId == 57 || trainingId == 58; // Y
            case 13, 14 -> trainingId == 59 || trainingId == 60;
            case 15, 16 -> trainingId == 61 || trainingId == 62; // Y
            case 17, 18 -> trainingId == 63 || trainingId == 64; // y
            case 19, 20 -> trainingId == 65 || trainingId == 66; // Y
            case 21, 22 -> trainingId == 87 || trainingId == 88; // y
            default -> false;
        };
    }

    // Error rates check
    static boolean verificationCheck(int testingId, int trainingId) {
        return switch (trainingId) {
            case 47, 48 -> testingId == 1 || testingId == 2;
            case 49, 50 -> testingId == 3 || testingId == 4;
            case 51, 52 -> testingId == 5 || testingId == 6;
            case 53, 54 -> testingId == 7 || testingId == 8;
            case 55, 56 -> testingId == 9 || testingId == 10;
            case 57, 58 -> testingId == 11 || testingId == 12;
            case 59, 60 -> testingId == 13 || testingId == 14;
            case 61, 62 -> testingId == 15 || testingId == 16;
            case 63, 64 -> testingId == 17 || testingId == 18;
            case 65, 66 -> testingId == 19 || testingId == 20;
            case 87, 88 -> testingId == 21 || testingId == 22;
            default -> false;
        };
    }
}