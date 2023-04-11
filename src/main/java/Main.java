import ai.djl.Application;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.BufferedImageFactory;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.Joints;
import ai.djl.repository.zoo.Criteria;
import ai.djl.translate.TranslateException;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.feature.DoubleFVComparison;
import org.openimaj.feature.FeatureVector;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.colour.ColourSpace;
import org.openimaj.image.colour.RGBColour;
import org.openimaj.image.connectedcomponent.GreyscaleConnectedComponentLabeler;
import org.openimaj.image.pixel.ConnectedComponent;
import org.openimaj.image.pixel.Pixel;
import org.openimaj.image.pixel.PixelSet;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.image.processor.PixelProcessor;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.ml.pca.FeatureVectorPCA;
import org.openimaj.util.pair.IntFloatPair;

import javax.imageio.ImageIO;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.atomic.AtomicReference;

public class Main {
    private static final String PATH = Paths.get("").toAbsolutePath() + "\\src\\main\\java\\";
    private static final Float[][] colours = new Float[][]{RGBColour.RED, RGBColour.ORANGE, RGBColour.YELLOW, RGBColour.GREEN, RGBColour.CYAN, RGBColour.BLUE, RGBColour.MAGENTA};
    private static Predictor<Image, Joints> predictor;

    public static void main(String[] args) throws IOException, TranslateException {
        AtomicReference<VFSListDataset<MBFImage>> training = new AtomicReference<>(new VFSListDataset<>(PATH + "biometrics\\training", ImageUtilities.MBFIMAGE_READER));
        AtomicReference<VFSListDataset<MBFImage>> testing = new AtomicReference<>(new VFSListDataset<>(PATH + "biometrics\\testing", ImageUtilities.MBFIMAGE_READER));

        ArrayList<ComputedImage> trainingImages = new ArrayList<>();
        ArrayList<ComputedImage> testingImages = new ArrayList<>();

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

        // Read and print the training images
        int count = 1;
        for (MBFImage trainingImage : training.get()) {
            trainingImages.add(readImage(trainingImage, count, true));
            count++;
        }

        // Read and print the testing images
        count = 1;
        for (MBFImage testingImage : testing.get()) {
            testingImages.add(readImage(testingImage, count, false));
            count++;
        }

        // Creates feature vectors from each image
        ArrayList<FeatureVector> featureVectors = new ArrayList<>();

        for (ComputedImage trainingImage : trainingImages) {
            trainingImage.extractFeature();
            featureVectors.add(trainingImage.getExtractedFeature());
        }
        for (ComputedImage testingImage : testingImages) {
            testingImage.extractFeature();
        }

        // Learning PCA basis
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
                } else if (furthestDistance == -1 || distance > furthestDistance) {
                    furthestDistance = distance;
                }
            }

            // Checks classification accuracy
            if (nearestImage != null && classificationCheck(testingImage.getId(), nearestImage.getId())) {
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
                    double distance = DoubleFVComparison.EUCLIDEAN.compare(trainingImageA.getExtractedFeature(), trainingImageB.getExtractedFeature());

                    if (distancesCheck(trainingImageA.getId(), trainingImageB.getId())) {
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

            if (FAR == FFR || smallestDistance == -1 || Math.abs(FAR - FFR) < smallestDistance) {
                EER = FAR * 100f;
                smallestDistance = Math.abs(FAR - FFR);

                if (FAR == FFR) {
                    break;
                }
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

        System.out.println(results);
    }

    static ComputedImage readImage(MBFImage image, int count, boolean isTraining) throws IOException, TranslateException {
        // Crop the image
        image = image.extractCenter((image.getWidth() / 2) + 100, (image.getHeight() / 2) + 115, 740, 1280);
        image.processInplace(new ResizeProcessor(0.5f));
        MBFImage clonedImage = image.clone();
        image = ColourSpace.convert(image, ColourSpace.CIE_Lab);

        // Get the pixel data
        float[][] imageData = image.getPixelVectorNative(new float[image.getWidth() * image.getHeight()][3]);

        // Groups the pixels into their classes
        FloatKMeans cluster = FloatKMeans.createExact(2);
        FloatCentroidsResult result = cluster.cluster(imageData);
        float[][] centroids = result.centroids;

        // Assigns pixels to a class
        image.processInplace((PixelProcessor<Float[]>) pixel -> {
            HardAssigner<float[], float[], IntFloatPair> assigner = result.defaultHardAssigner();

            float[] set1 = new float[3];
            for (int i = 0; i < 3; i++) {
                set1[i] = pixel[i];
            }
            float[] centroid = centroids[assigner.assign(set1)];

            Float[] set2 = new Float[3];
            for (int i = 0; i < 3; i++) {
                set2[i] = centroid[i];
            }
            return set2;
        });
        image = ColourSpace.convert(image, ColourSpace.RGB);

        // Get the two connected components
        GreyscaleConnectedComponentLabeler labeler = new GreyscaleConnectedComponentLabeler();
        List<ConnectedComponent> components = labeler.findComponents(image.flatten());

        // Get the person component
        components.sort(Comparator.comparingInt(PixelSet::calculateArea));
        Collections.reverse(components);
        ConnectedComponent component = components.get(1);

        // Get the boundary pixels and all contained pixels
        Set<Pixel> pixels = component.getPixels();

        // Remove all unnecessary pixels from image
        for (int y = 0; y < clonedImage.getHeight(); y++) {
            for (int x = 0; x < clonedImage.getWidth(); x++) {
                if (!pixels.contains(new Pixel(x, y))) {
                    clonedImage.getBand(0).pixels[y][x] = 1;
                    clonedImage.getBand(1).pixels[y][x] = 1;
                    clonedImage.getBand(2).pixels[y][x] = 1;
                }
            }
        }

        // Print the original image
        String resultPath = isTraining ? "training" : "testing";
        File imageFile = new File(PATH + "computed\\" + resultPath + "\\" + count + ".jpg");
        ImageUtilities.write(clonedImage, imageFile);

        // Find the joints of training images
        File jointsImageFile = new File(PATH + "joints\\" + resultPath + "\\" + count + ".jpg");
        Image jointsImage = BufferedImageFactory.getInstance().fromImage(ImageIO.read(imageFile));
        Joints joints = predictor.predict(jointsImage);

        int countColour = 0;
        for (Joints.Joint joint : joints.getJoints()) {
            Pixel pixel = new Pixel((int) (joint.getX() * clonedImage.getWidth()), (int) (joint.getY() * clonedImage.getHeight()));

            clonedImage.drawPoint(pixel, colours[countColour], 5);
            clonedImage.drawPolygon(component.toPolygon(), RGBColour.RED);

            if (countColour < colours.length - 1) {
                countColour++;
            } else {
                countColour = 0;
            }
        }
        clonedImage.drawPoint(component.calculateCentroidPixel(), RGBColour.RED, 5);
        clonedImage.drawPolygon(component.toPolygon(), RGBColour.RED);

        clonedImage = clonedImage.extractROI(component.calculateRegularBoundingBox());
        ImageUtilities.write(clonedImage, jointsImageFile);
        return new ComputedImage(count, component, joints);
    }

    // Classification checks
    static boolean exactClassificationCheck(int testingId, int trainingId) {
        return switch (testingId) {
            case 1 -> trainingId == 48;
            case 2 -> trainingId == 47;
            case 3 -> trainingId == 50;
            case 4 -> trainingId == 49;
            case 5 -> trainingId == 52;
            case 6 -> trainingId == 51;
            case 7 -> trainingId == 54;
            case 8 -> trainingId == 53;
            case 9 -> trainingId == 56;
            case 10 -> trainingId == 55;
            case 11 -> trainingId == 58;
            case 12 -> trainingId == 57;
            case 13 -> trainingId == 60;
            case 14 -> trainingId == 59;
            case 15 -> trainingId == 62;
            case 16 -> trainingId == 61;
            case 17 -> trainingId == 64;
            case 18 -> trainingId == 63;
            case 19 -> trainingId == 66;
            case 20 -> trainingId == 65;
            case 21 -> trainingId == 88;
            case 22 -> trainingId == 87;
            default -> false;
        };
    }

    static boolean classificationCheck(int testingId, int trainingId) {
        return switch (testingId) {
            case 1, 2 -> trainingId == 47 || trainingId == 48;
            case 3, 4 -> trainingId == 49 || trainingId == 50;
            case 5, 6 -> trainingId == 51 || trainingId == 52;
            case 7, 8 -> trainingId == 53 || trainingId == 54;
            case 9, 10 -> trainingId == 55 || trainingId == 56;
            case 11, 12 -> trainingId == 57 || trainingId == 58;
            case 13, 14 -> trainingId == 59 || trainingId == 60;
            case 15, 16 -> trainingId == 61 || trainingId == 62;
            case 17, 18 -> trainingId == 63 || trainingId == 64;
            case 19, 20 -> trainingId == 65 || trainingId == 66;
            case 21, 22 -> trainingId == 87 || trainingId == 88;
            default -> false;
        };
    }

    static boolean distancesCheck(int testingId, int trainingId) {
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