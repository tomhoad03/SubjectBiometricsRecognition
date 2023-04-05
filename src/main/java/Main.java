import ai.djl.Application;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.BufferedImageFactory;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.Joints;
import ai.djl.repository.zoo.Criteria;
import ai.djl.translate.TranslateException;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.DoubleFVComparison;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.colour.ColourSpace;
import org.openimaj.image.connectedcomponent.GreyscaleConnectedComponentLabeler;
import org.openimaj.image.pixel.ConnectedComponent;
import org.openimaj.image.pixel.Pixel;
import org.openimaj.image.pixel.PixelSet;
import org.openimaj.image.processing.convolution.FFastGaussianConvolve;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.image.processor.PixelProcessor;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.util.pair.IntFloatPair;

import javax.imageio.ImageIO;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.atomic.AtomicReference;

public class Main {
    private static final String PATH = Paths.get("").toAbsolutePath() + "\\src\\main\\java\\";
    private static final float SPEED_FACTOR = 0.25f; // 1f - Normal running, 0.25f - Fast running
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
            ComputedImage image = readImage(trainingImage, count, true);
            if (count % 2 == 1) {
                trainingImagesFront.add(image);
            } else {
                trainingImagesSide.add(image);
            }
            count++;
        }

        // Read and print the testing images
        count = 1;
        for (MBFImage testingImage : testing.get()) {
            ComputedImage image = readImage(testingImage, count, false);
            if (count % 2 == 0) {
                testingImagesFront.add(image);
            } else {
                testingImagesSide.add(image);
            }
            count++;
        }

        double correctClassificationCountFront = classifyImages(trainingImagesFront, testingImagesFront);
        double correctClassificationCountSide = classifyImages(trainingImagesSide, testingImagesSide);

        // Print the results
        System.out.println("Finished!"
                + "\n" + "Front Classification Accuracy = " + ((correctClassificationCountFront / 11f) * 100f) + "%"
                + "\n" + "Side Classification Accuracy = " + ((correctClassificationCountSide / 11f) * 100f) + "%"
                + "\n" + "Correct Classification Rate (CCR) = " + (((correctClassificationCountFront + correctClassificationCountSide) / 22f) * 100f) + "%");
    }

    static ComputedImage readImage(MBFImage image, int count, boolean isTraining) throws IOException, TranslateException {
        // Crop the image
        image = image.extractCenter((image.getWidth() / 2) + 80, (image.getHeight() / 2) + 110, 750, 1280);
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
        ImageUtilities.write(clonedImage, imageFile);
        Image jointsImage = BufferedImageFactory.getInstance().fromImage(ImageIO.read(imageFile));

        Joints joints = predictor.predict(jointsImage);
        jointsImage.drawJoints(joints);
        jointsImage.save(Files.newOutputStream(Paths.get(PATH + "joints\\" + resultPath + "\\" + count + ".png")), "png");

        return new ComputedImage(count,
                clonedImage, // The image (to be removed later)
                component.calculateCentroidPixel(), // The persons centroid
                component.calculateBoundaryDistanceFromCentre().toArray(), // Boundary distances
                new DoubleFV(component.calculateConvexHull().calculateSecondMomentCentralised()).normaliseFV(), // Second order centralised moment
                joints); // Joint positions
    }

    // Classifies the dataset
    static double classifyImages(ArrayList<ComputedImage> trainingImages, ArrayList<ComputedImage> testingImages) {
        // Trains the assigner
        for (ComputedImage trainingImage : trainingImages) {
            trainingImage.setExtractedFeature(extractSilhouetteFV(trainingImage).concatenate(extractJointsFV(trainingImage)));
        }

        for (ComputedImage testingImage : testingImages) {
            testingImage.setExtractedFeature(extractSilhouetteFV(testingImage).concatenate(extractJointsFV(testingImage)));
        }

        // Nearest neighbour to find the closest training image to each testing image
        float correctClassificationCount = 0f;

        for (ComputedImage testingImage : testingImages) {
            ComputedImage nearestImage = null;
            double nearestDistance = -1;

            // K-Nearest Neighbours based on second order centralised moment
            trainingImages.sort(Comparator.comparingDouble(o -> DoubleFVComparison.EUCLIDEAN.compare(o.getSecondCentralisedMoment(), testingImage.getSecondCentralisedMoment())));
            List<ComputedImage> kNearestTrainingImages = trainingImages.subList(0, trainingImages.size() / 2);

            // Finds the nearest image
            for (ComputedImage trainingImage : kNearestTrainingImages) {
                double distance = DoubleFVComparison.EUCLIDEAN.compare(trainingImage.getExtractedFeature(), testingImage.getExtractedFeature());

                if (nearestDistance == -1 || distance < nearestDistance) {
                    nearestDistance = distance;
                    nearestImage = trainingImage;
                }
            }

            // Checks classification accuracy
            if (nearestImage != null && classificationTest(testingImage.getId(), nearestImage.getId())) {
                DisplayUtilities.display(testingImage.getImage());
                DisplayUtilities.display(nearestImage.getImage());
                correctClassificationCount += 1f;
            }
        }
        return correctClassificationCount;
    }

    // Extract silhouette feature vector
    static DoubleFV extractSilhouetteFV(ComputedImage image) {
        float[] floatDistances = image.getBoundaryDistances();

        ArrayList<Double> listDistances = new ArrayList<>();
        for (float value : floatDistances) {
            listDistances.add((double) value);
        }
        listDistances.sort(Double::compare);

        double[] doubleDistances = new double[5000];
        for (int i = 4999; i >= 0; i--) {

            try {
                doubleDistances[i] = listDistances.get(i);
            } catch (Exception e) {
                doubleDistances[i] = 0;
            }
        }
        return new DoubleFV(doubleDistances).normaliseFV();
    }

    // Extract joints feature vector
    static DoubleFV extractJointsFV(ComputedImage image) {
        List<Joints.Joint> joints = image.getJoints().getJoints();
        ArrayList<Double> jointRadii = new ArrayList<>();
        double centroidX = image.getCentroid().getX() / image.getImage().getWidth(), centroidY =  image.getCentroid().getY() / image.getImage().getHeight();

        for (Joints.Joint joint : joints) {
            double xDiff = joint.getX() - centroidX, yDiff = joint.getY() - centroidY;
            double radius = Math.sqrt(Math.pow(xDiff, 2) + Math.pow(yDiff, 2));
            jointRadii.add(radius);
        }
        jointRadii.sort(Double::compare);

        double[] array1 = new double[17];
        for (int i = 16; i >= 0; i--) {
            try {
                array1[i] = jointRadii.get(i);
            } catch (Exception e) {
                array1[i] = 0;
            }
        }
        return new DoubleFV(array1).normaliseFV();
    }

    // CCR test - not used in classification
    static boolean classificationTest(int testingId, int trainingId) {
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
}