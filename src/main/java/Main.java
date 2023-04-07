import ai.djl.Application;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.BufferedImageFactory;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.Joints;
import ai.djl.repository.zoo.Criteria;
import ai.djl.translate.TranslateException;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.feature.DoubleFVComparison;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.colour.ColourSpace;
import org.openimaj.image.colour.RGBColour;
import org.openimaj.image.connectedcomponent.GreyscaleConnectedComponentLabeler;
import org.openimaj.image.pixel.ConnectedComponent;
import org.openimaj.image.pixel.Pixel;
import org.openimaj.image.pixel.PixelSet;
import org.openimaj.image.processing.convolution.FFastGaussianConvolve;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.image.processor.PixelProcessor;
import org.openimaj.image.typography.hershey.HersheyFont;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.util.pair.IntFloatPair;

import javax.imageio.ImageIO;
import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.atomic.AtomicReference;

public class Main {
    private static final String PATH = Paths.get("").toAbsolutePath() + "\\src\\main\\java\\";
    private static final Float[][] colours = new Float[][]{RGBColour.RED, RGBColour.ORANGE, RGBColour.YELLOW, RGBColour.GREEN, RGBColour.CYAN, RGBColour.BLUE, RGBColour.MAGENTA};
    private static final String[] bodyParts = new String[]{"Nose", "Right Eye", "Left Eye", "Right Ear", "Left Ear", "Right Shoulder", "Left Shoulder", "Right Elbow", "Left Elbow", "Right Hand", "Left Hand", "Right Hip", "Left Hip", "Right Knee", "Left Knee", "Right Foot", "Left Foot"};
    private static final float SPEED_FACTOR = 1f; // 1f - Normal running, 0.25f - Fast running
    private static Predictor<Image, Joints> predictor;

    public static void main(String[] args) throws IOException, TranslateException {
        AtomicReference<VFSListDataset<MBFImage>> training = new AtomicReference<>(new VFSListDataset<>(PATH + "biometrics\\training", ImageUtilities.MBFIMAGE_READER));
        AtomicReference<VFSListDataset<MBFImage>> testing = new AtomicReference<>(new VFSListDataset<>(PATH + "biometrics\\testing", ImageUtilities.MBFIMAGE_READER));

        ArrayList<ComputedImage> trainingImagesFront = new ArrayList<>();
        ArrayList<ComputedImage> trainingImagesSide = new ArrayList<>();

        ArrayList<ComputedImage> testingImagesFront = new ArrayList<>();
        ArrayList<ComputedImage> testingImagesSide = new ArrayList<>();

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
            if (count % 2 == 1) {
                trainingImagesFront.add(readImage(trainingImage, count, true, true));
            } else {
                trainingImagesSide.add(readImage(trainingImage, count, true, false));
            }
            count++;
        }

        // Read and print the testing images
        count = 1;
        for (MBFImage testingImage : testing.get()) {
            if (count % 2 == 0) {
                testingImagesFront.add(readImage(testingImage, count, false, true));
            } else {
                testingImagesSide.add(readImage(testingImage, count, false, false));
            }
            count++;
        }

        double[] frontClassificationResults = classifyImages(trainingImagesFront, testingImagesFront);
        double[] sideClassificationResults = classifyImages(trainingImagesSide, testingImagesSide);

        // Print the results
        System.out.println("Front Classification Accuracy = " + (float) ((frontClassificationResults[0] / 11f) * 100f) + "%"
                + "\n" + "Side Classification Accuracy = " + (float) ((sideClassificationResults[0] / 11f) * 100f) + "%"
                + "\n" + "Front Incorrect Accuracy: " + (float) ((frontClassificationResults[1] / (22f - frontClassificationResults[0])) * 100f) + "%"
                + "\n" + "Side Incorrect Accuracy: " + (float) ((sideClassificationResults[1] / (22f - sideClassificationResults[0])) * 100f) + "%"
                + "\n" + "Correct Classification Rate (CCR) = " + (float) (((frontClassificationResults[0] + sideClassificationResults[0]) / 22f) * 100f) + "%");
    }

    static ComputedImage readImage(MBFImage image, int count, boolean isTraining, boolean isFront) throws IOException, TranslateException {
        // Crop the image
        image = image.extractCenter((image.getWidth() / 2) + 100, (image.getHeight() / 2) + 115, 740, 1280);
        image.processInplace(new ResizeProcessor(SPEED_FACTOR));
        MBFImage clonedImage = image.clone();

        // Apply a Gaussian blur to reduce noise
        image = ColourSpace.convert(image, ColourSpace.CIE_Lab);
        image.processInplace(new FFastGaussianConvolve(2, 2));

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

        // Find the joints of training images
        String resultPath = isTraining ? "training" : "testing";
        File imageFile = new File(PATH + "computed\\" + resultPath + "\\" + count + ".jpg");
        File jointsImageFile = new File(PATH + "joints\\" + resultPath + "\\" + count + ".jpg");
        ImageUtilities.write(clonedImage, imageFile);

        Image jointsImage = BufferedImageFactory.getInstance().fromImage(ImageIO.read(imageFile));
        Joints joints = predictor.predict(jointsImage);

        int countColour = 0, bodyPartCount = 0;
        for (Joints.Joint joint : joints.getJoints()) {
            Pixel pixel = new Pixel((int) (joint.getX() * clonedImage.getWidth()), (int) (joint.getY() * clonedImage.getHeight()));

            clonedImage.drawPoint(pixel, colours[countColour], 5);
            clonedImage.drawText(bodyParts[bodyPartCount], pixel, HersheyFont.TIMES_MEDIUM, 20, RGBColour.RED);
            clonedImage.drawPolygon(component.toPolygon(), RGBColour.RED);

            if (countColour < colours.length - 1) {
                countColour++;
            } else {
                countColour = 0;
            }
            bodyPartCount++;
        }
        clonedImage = clonedImage.extractROI(component.calculateRegularBoundingBox());
        ImageUtilities.write(clonedImage, jointsImageFile);

        return new ComputedImage(count,
                clonedImage, // The image (to be removed later)
                isFront,
                component.calculateCentroidPixel(), // The persons centroid
                component.getOuterBoundary(), // Boundary pixels
                joints); // Joint positions
    }

    // Classifies the dataset
    static double[] classifyImages(ArrayList<ComputedImage> trainingImages, ArrayList<ComputedImage> testingImages) {
        // Trains the assigner
        for (ComputedImage trainingImage : trainingImages) {
            trainingImage.extractFeature();
        }
        for (ComputedImage testingImage : testingImages) {
            testingImage.extractFeature();
        }

        // Nearest neighbour to find the closest training image to each testing image
        float correctClassificationCount = 0f;
        double incorrectAccuracySum = 0f;

        for (ComputedImage testingImage : testingImages) {
            ComputedImage nearestImage = null;
            double nearestDistance = -1, furthestDistance = -1, correctDistance = 0f;

            // Finds the nearest image
            for (ComputedImage trainingImage : trainingImages) {
                double distance = DoubleFVComparison.EUCLIDEAN.compare(trainingImage.getExtractedFeature(), testingImage.getExtractedFeature());

                if (nearestDistance == -1 || distance < nearestDistance) {
                    nearestDistance = distance;
                    nearestImage = trainingImage;
                } else if (furthestDistance == -1 || distance > furthestDistance) {
                    furthestDistance = distance;
                }

                if (classificationTest(testingImage.getId(), trainingImage.getId())) {
                    correctDistance = distance;
                }
            }

            // Checks classification accuracy
            if (nearestImage != null && classificationTest(testingImage.getId(), nearestImage.getId())) {
                correctClassificationCount += 1f;
            } else {
                incorrectAccuracySum += ((correctDistance - nearestDistance) / (furthestDistance - correctDistance));
            }
        }
        return new double[]{correctClassificationCount, incorrectAccuracySum};
    }

    // CCR test - not used in classification
    static boolean classificationTest(int testingId, int trainingId) {
        return switch (testingId) {
            case 1 -> trainingId == 48; // n
            case 2 -> trainingId == 47; // n
            case 3 -> trainingId == 50; // n
            case 4 -> trainingId == 49; // n
            case 5 -> trainingId == 52; // y
            case 6 -> trainingId == 51; // n
            case 7 -> trainingId == 54; // n
            case 8 -> trainingId == 53; // n
            case 9 -> trainingId == 56; // y
            case 10 -> trainingId == 55; // n
            case 11 -> trainingId == 58; // y
            case 12 -> trainingId == 57; // n
            case 13 -> trainingId == 60; // y
            case 14 -> trainingId == 59; // n
            case 15 -> trainingId == 62; // y
            case 16 -> trainingId == 61; // n
            case 17 -> trainingId == 64; // n
            case 18 -> trainingId == 63; // n
            case 19 -> trainingId == 66; // y
            case 20 -> trainingId == 65; // n
            case 21 -> trainingId == 88; // n
            case 22 -> trainingId == 87; // y
            default -> false;
        };
    }
}