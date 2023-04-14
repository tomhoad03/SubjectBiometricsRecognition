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
import org.openimaj.image.segmentation.KMSpatialColourSegmenter;
import org.openimaj.image.segmentation.SegmentationUtilities;
import org.openimaj.math.geometry.shape.Rectangle;
import org.openimaj.ml.pca.FeatureVectorPCA;

import javax.imageio.ImageIO;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.atomic.AtomicReference;

public class Main {
    private static final String PATH = Paths.get("").toAbsolutePath() + "\\src\\main\\java\\";
    private static Predictor<Image, Joints> predictor;
    private static final Float[][] temperatures = new Float[48][];

    /**
     * Runs the classification and prints the results
     */
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

        // Reads the training images
        int count = 1;
        for (MBFImage trainingImage : training.get()) {
            trainingImages.add(readImage(trainingImage, count, true));
            System.out.print("|");
            count++;
        }

        // Reads the testing images
        count = 1;
        for (MBFImage testingImage : testing.get()) {
            testingImages.add(readImage(testingImage, count, false));
            System.out.print("|");
            count++;
        }

        // Learning the PCA basis
        for (ComputedImage trainingImage : trainingImages) {
            featureVectors.add(trainingImage.getExtractedFeature());
        }
        FeatureVectorPCA pca = new FeatureVectorPCA();
        pca.learnBasis(featureVectors);

        // Nearest neighbour to find the closest training image to each testing image
        float correctCount = 0f;
        ArrayList<Double> intraDistances = new ArrayList<>(), interDistances = new ArrayList<>();
        StringBuilder histogram = new StringBuilder("idA,idB,distance,type\n");

        for (ComputedImage testingImage : testingImages) {
            ComputedImage nearestImage = null;
            double nearestDistance = -1;

            // Finds the nearest image
            for (ComputedImage trainingImage : trainingImages) {
                double distance = DoubleFVComparison.EUCLIDEAN.compare(pca.project(trainingImage.getExtractedFeature()), pca.project(testingImage.getExtractedFeature()));

                // Checks if it's the nearest or furthest distance
                if (nearestDistance == -1 || distance < nearestDistance) {
                    nearestDistance = distance;
                    nearestImage = trainingImage;
                }
            }

            // Checks if the classification is correct
            if (nearestImage != null) {
                if (classificationCheck(testingImage.getId(), nearestImage.getId())) {
                    correctCount += 1f;
                }
                histogram.append(testingImage.getId()).append(",").append(nearestImage.getId()).append(",").append(nearestDistance).append(",inter\n");
                interDistances.add(nearestDistance);
            }
        }

        // Calculating the histogram of distances
        double correctClassificationRate = (correctCount / 22f) * 100f;

        for (int i = 0; i < trainingImages.size(); i++) {
            for (int j = i; j < trainingImages.size(); j++) {
                ComputedImage trainingImageA = trainingImages.get(i);
                ComputedImage trainingImageB = trainingImages.get(j);

                if (trainingImageA.getId() != trainingImageB.getId()) {
                    double distance = DoubleFVComparison.EUCLIDEAN.compare(pca.project(trainingImageA.getExtractedFeature()), pca.project(trainingImageB.getExtractedFeature()));
                    histogram.append(trainingImageA.getId()).append(",").append(trainingImageB.getId()).append(",").append(distance).append(",intra\n");
                    intraDistances.add(distance);
                }
            }
        }

        // Sort the distances
        interDistances.sort(Comparator.comparingDouble(o -> o));
        intraDistances.sort(Comparator.comparingDouble(o -> o));
        double EER = 0f, smallestDistance = -1f, finalThreshold = 0f;

        // Equal error rate calculation
        for (double threshold = 0f; threshold < 1f; threshold += 0.000001f) {
            double tempThreshold = threshold;
            double FAR = interDistances.stream().filter(a -> a > tempThreshold).count() / (double) interDistances.size();
            double FRR = intraDistances.stream().filter(a -> a < tempThreshold).count() / (double) intraDistances.size();

            if (FAR == FRR || smallestDistance == -1f || Math.abs(FAR - FRR) < smallestDistance) {
                EER = FAR * 100f;
                smallestDistance = Math.abs(FAR - FRR);
                finalThreshold = tempThreshold;
            }
            if (FAR == FRR) {
                break;
            }
        }
        long endTime = System.currentTimeMillis();

        // Prints the CCR and EER
        String results = "Correct Classification Rate (CCR) = " + (float) correctClassificationRate + "%"
                + "\n" + "Equal Error Rate: " + (float) EER + "%"
                + "\n" + "EER Threshold: " + (float) finalThreshold
                + "\n" + "Duration: " + (endTime - startTime) + "ms";

        File resultsFile = new File(PATH + "\\results.txt");
        FileWriter fileWriter = new FileWriter(resultsFile);
        fileWriter.write(results);
        fileWriter.close();

        // Prints the histogram of distances
        File histogramFile = new File(PATH + "\\histogram.csv");
        FileWriter histogramWriter = new FileWriter(histogramFile);
        histogramWriter.write(histogram.toString());
        histogramWriter.close();

        System.out.println("\n" + results);
    }

    static ComputedImage readImage(MBFImage image, int count, boolean isTraining) throws IOException, TranslateException {
        // Crop the image
        image = image.extractCenter((image.getWidth() / 2) + 100, (image.getHeight() / 2) + 115, 740, 1280);
        image.processInplace(new ResizeProcessor(1f));
        MBFImage segmentedImage = image.clone();

        // Image segmentation
        KMSpatialColourSegmenter segmenter = new KMSpatialColourSegmenter(ColourSpace.CIE_Lab, 2);
        SegmentationUtilities.renderSegments(image, segmenter.segment(image));

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
        for (int y = 0; y < segmentedImage.getHeight(); y++) {
            for (int x = 0; x < segmentedImage.getWidth(); x++) {
                if (!pixels.contains(new Pixel(x, y))) {
                    segmentedImage.getBand(0).pixels[y][x] = 1;
                    segmentedImage.getBand(1).pixels[y][x] = 1;
                    segmentedImage.getBand(2).pixels[y][x] = 1;
                }
            }
        }

        // Calculates bounding box metrics
        MBFImage temperatureImage = segmentedImage.clone();
        Rectangle boundingBox = component.calculateRegularBoundingBox();
        Pixel centroid = component.calculateCentroidPixel();
        double[] temperatureCounts = new double[48];

        // Creates the temperature image
        for (int y = 0; y < temperatureImage.getHeight(); y++) {
            for (int x = 0; x < temperatureImage.getWidth(); x++) {
                if (pixels.contains(new Pixel(x, y))) {
                    double divide = ((float) temperatureImage.getHeight() - (float) y) / boundingBox.getHeight();
                    double doubleIndex = (divide * temperatures.length) / 2f;
                    int index = (int) Math.floor(doubleIndex);

                    // Splits the regions into left and right
                    if (x > centroid.getX()) {
                        index += temperatures.length / 2f;
                    }

                    // Sets the temperature of the pixel
                    try {
                        Float[] temperature = temperatures[index];
                        temperatureImage.getBand(0).pixels[y][x] = temperature[0];
                        temperatureImage.getBand(1).pixels[y][x] = temperature[1];
                        temperatureImage.getBand(2).pixels[y][x] = temperature[2];

                        if (temperatureCounts[index] == 0) {
                            temperatureCounts[index] = 1;
                        } else {
                            temperatureCounts[index] = temperatureCounts[index] + 1;
                        }
                    } catch (Exception ignored) { }
                }
            }
        }

        // Print the segmented image
        String resultPath = isTraining ? "training" : "testing";
        File imageFile = new File(PATH + "segmented\\" + resultPath + "\\" + count + ".jpg");
        ImageUtilities.write(segmentedImage, imageFile);

        // Print the temperature image
        File temperatureImageFile = new File(PATH + "temperature\\" + resultPath + "\\" + count + ".jpg");
        ImageUtilities.write(temperatureImage.extractROI(boundingBox), temperatureImageFile);

        // Print the joints image
        File jointsImageFile = new File(PATH + "joints\\" + resultPath + "\\" + count + ".jpg");
        Image jointsImage = BufferedImageFactory.getInstance().fromImage(ImageIO.read(imageFile));
        Joints joints = predictor.predict(jointsImage);

        // Find the joints from the segmented image
        for (Joints.Joint joint : joints.getJoints()) {
            Pixel pixel = new Pixel((int) (joint.getX() * segmentedImage.getWidth()), (int) (joint.getY() * segmentedImage.getHeight()));
            segmentedImage.drawPoint(pixel, RGBColour.RED, 7);
        }

        // Draw the centroid point and component outline
        segmentedImage.drawPoint(component.calculateCentroidPixel(), RGBColour.BLUE, 7);
        segmentedImage.drawPolygon(component.toPolygon(), 3, RGBColour.GREEN);
        segmentedImage = segmentedImage.extractROI(boundingBox);
        ImageUtilities.write(segmentedImage, jointsImageFile);

        return new ComputedImage(count, component, joints, temperatureCounts);
    }

    /**
     * Classification check
     * @param testingId Testing FV id
     * @param trainingId Training FV id
     * @return True if correct classification
     */
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
}