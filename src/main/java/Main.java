import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.feature.ByteFV;
import org.openimaj.feature.ByteFVComparison;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.SparseIntFVComparison;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.colour.ColourSpace;
import org.openimaj.image.connectedcomponent.GreyscaleConnectedComponentLabeler;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.engine.DoGSIFTEngine;
import org.openimaj.image.feature.local.keypoints.Keypoint;
import org.openimaj.image.pixel.ConnectedComponent;
import org.openimaj.image.pixel.Pixel;
import org.openimaj.image.pixel.PixelSet;
import org.openimaj.image.processing.convolution.FFastGaussianConvolve;
import org.openimaj.image.processor.PixelProcessor;
import org.openimaj.ml.clustering.FeatureVectorCentroidsResult;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FeatureVectorKMeans;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.util.pair.IntFloatPair;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.*;

public class Main {
    private static final int SAMPLE_SIZE = 22;
    private static final int NUMBER_OF_FEATURE_CLUSTERS = 500;
    private static final int NUMBER_OF_HISTOGRAM_CLUSTERS = 44;
    private static ArrayList<ComputedImage> trainingImages = new ArrayList<>();
    private static ArrayList<ComputedImage> testingImages = new ArrayList<>();

    public static void main(String[] args) throws IOException {
        final VFSListDataset<MBFImage> training = new VFSListDataset<>(Paths.get("").toAbsolutePath() + "\\src\\main\\java\\biometrics\\training", ImageUtilities.MBFIMAGE_READER);
        final VFSListDataset<MBFImage> testing = new VFSListDataset<>(Paths.get("").toAbsolutePath() + "\\src\\main\\java\\biometrics\\testing", ImageUtilities.MBFIMAGE_READER);

        // Read training images
        System.out.println("Reading training images...");
        int count = 1;

        for (MBFImage trainingImage : training) {
            trainingImages.add(computeImage(trainingImage, count));
            count++;
        }

        // Read the testing images
        System.out.println("Reading testing images...");
        count = 1;

        for (MBFImage testingImage : testing) {
            testingImages.add(computeImage(testingImage, count));
            count++;
        }

        for (int i = 50; i < 1000; i += 50) {
            runClassification(SAMPLE_SIZE, i, NUMBER_OF_HISTOGRAM_CLUSTERS);
        }
    }


    static void runClassification(int sampleSize, int featureClusters, int histogramClusters) {
        // Trains the assigner from a training sample
        List<ComputedImage> subTrainingImages = trainingImages.subList(0, sampleSize);
        List<ByteFV> featuresList = new ArrayList<>();
        DoGSIFTEngine engine = new DoGSIFTEngine();

        // Sample the training dataset
        // System.out.println("Sampling training images...");

        for (ComputedImage trainingImage : subTrainingImages) {
            featuresList.addAll(extractFeatures(engine, trainingImage));
        }

        // K-Means clusters sampled features
        FeatureVectorKMeans<ByteFV> kMeans = FeatureVectorKMeans.createExact(featureClusters, ByteFVComparison.EUCLIDEAN);
        FeatureVectorCentroidsResult<ByteFV> result = kMeans.cluster(featuresList);
        HardAssigner<ByteFV, float[], IntFloatPair> assigner = result.defaultHardAssigner();
        ArrayList<SparseIntFV> extractedTrainingFeatures = new ArrayList<>();

        // Creates a BoVW for each training image
        // System.out.println("Training classifier...");

        for (ComputedImage trainingImage : trainingImages) {
            BagOfVisualWords bagOfVisualWords = new BagOfVisualWords(assigner);
            SparseIntFV extractedFeature = bagOfVisualWords.aggregateVectorsRaw(extractFeatures(engine, trainingImage));

            trainingImage.setExtractedFeature(extractedFeature);
            extractedTrainingFeatures.add(extractedFeature);
        }

        // K-Means clusters the bags of visual words
        FeatureVectorKMeans<SparseIntFV> kMeans2 = FeatureVectorKMeans.createExact(histogramClusters, SparseIntFVComparison.EUCLIDEAN);
        FeatureVectorCentroidsResult<SparseIntFV> result2 = kMeans2.cluster(extractedTrainingFeatures);
        SparseIntFV[] centroids = result2.centroids;
        HardAssigner<SparseIntFV, float[], IntFloatPair> assigner2 = result2.defaultHardAssigner();

        // Finds the closest centroid for each feature vector
        for (ComputedImage trainingImage : trainingImages) {
            trainingImage.setCentroid(centroids[assigner2.assign(trainingImage.getExtractedFeature())]);
        }

        // Creates a BoVW for each testing image
        // System.out.println("Classifying testing...");

        for (ComputedImage testingImage : testingImages) {
            BagOfVisualWords bagOfVisualWords = new BagOfVisualWords(assigner);
            testingImage.setExtractedFeature(bagOfVisualWords.aggregateVectorsRaw(extractFeatures(engine, testingImage)));
            testingImage.setCentroid(centroids[assigner2.assign(testingImage.getExtractedFeature())]);
        }

        // Identify the testing and training images with the same centroid
        float correctClassificationCount = 0f;

        for (ComputedImage testingImage : testingImages) {
            for (ComputedImage trainingImage : trainingImages) {
                if (trainingImage.getCentroid() == testingImage.getCentroid()) {
                    DisplayUtilities.display(trainingImage.getImage());

                    if (classificationTest(testingImage.getId(), trainingImage.getId())) {
                        correctClassificationCount += 1f;
                    }
                }
            }

            DisplayUtilities.display(testingImage.getImage());
        }

        // Print the results
        System.out.println("Finished!" + "\n" + "(" + sampleSize + "/" + featureClusters + "/" + histogramClusters + ")" + "\n" + "Classification accuracy = " + ((correctClassificationCount / 22f) * 100f) + "%");
    }

    static ComputedImage computeImage(MBFImage image, int count) {
        // Resize and crop the image
        image = image.extractCenter(image.getWidth() / 2, (image.getHeight() / 2) + 120, 720, 1260);
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
        ConnectedComponent personComponent = components.get(1);

        // Get the boundary pixels and all contained pixels
        Set<Pixel> pixels = personComponent.getPixels();

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

        return new ComputedImage(count, true, personComponent, clonedImage);
    }

    // Extracts SIFT descriptors from image (invariant to S+T+R)
    static ArrayList<ByteFV> extractFeatures(DoGSIFTEngine engine, ComputedImage image) {
        ArrayList<ByteFV> featuresList = new ArrayList<>();
        LocalFeatureList<Keypoint> keypointList = engine.findFeatures(image.getImage().flatten());

        for (Keypoint keypoint : keypointList) {
            featuresList.add(keypoint.getFeatureVector());
        }
        return featuresList;
    }

    // Classification test
    private static boolean classificationTest(int testingId, int trainingId) {
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