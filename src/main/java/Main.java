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
    static int count = 1;

    public static void main(String[] args) throws IOException {
        final VFSListDataset<MBFImage> training = new VFSListDataset<>(Paths.get("").toAbsolutePath() + "\\src\\main\\java\\biometrics\\training", ImageUtilities.MBFIMAGE_READER);
        final VFSListDataset<MBFImage> testing = new VFSListDataset<>(Paths.get("").toAbsolutePath() + "\\src\\main\\java\\biometrics\\testing", ImageUtilities.MBFIMAGE_READER);

        ArrayList<ComputedImage> trainingImages = new ArrayList<>();
        ArrayList<ComputedImage> testingImages = new ArrayList<>();

        // Read training images
        for (MBFImage trainingImage : training) {
            System.out.println("Reading training... " + count);
            trainingImages.add(computeImage(trainingImage));
            count++;
        }
        count = 1;

        // Trains the assigner from a training sample
        List<ComputedImage> subTrainingImages = trainingImages.subList(0, 22);
        List<ByteFV> featuresList = new ArrayList<>();
        DoGSIFTEngine engine = new DoGSIFTEngine();

        for (ComputedImage trainingImage : subTrainingImages) {
            System.out.println("Sampling training... " + count);
            featuresList.addAll(extractFeatures(engine, trainingImage));
            count++;
        }
        count = 1;

        // K-Means clusters sampled features
        FeatureVectorKMeans<ByteFV> kMeans = FeatureVectorKMeans.createExact(500, ByteFVComparison.EUCLIDEAN);
        FeatureVectorCentroidsResult<ByteFV> result = kMeans.cluster(featuresList);
        HardAssigner<ByteFV, float[], IntFloatPair> assigner = result.defaultHardAssigner();
        ArrayList<SparseIntFV> extractedTrainingFeatures = new ArrayList<>();

        // Training the classifier
        for (ComputedImage trainingImage : trainingImages) {
            System.out.println("Training... " + count);

            // Creates a BoVW for the image
            BagOfVisualWords bagOfVisualWords = new BagOfVisualWords(assigner);
            SparseIntFV extractedFeature = bagOfVisualWords.aggregateVectorsRaw(extractFeatures(engine, trainingImage));
            trainingImage.setExtractedFeature(extractedFeature);
            extractedTrainingFeatures.add(extractedFeature);
            count++;
        }
        count = 1;

        // K-Means clusters bags of visual words
        FeatureVectorKMeans<SparseIntFV> kMeans2 = FeatureVectorKMeans.createExact(44, SparseIntFVComparison.EUCLIDEAN);
        FeatureVectorCentroidsResult<SparseIntFV> result2 = kMeans2.cluster(extractedTrainingFeatures);
        SparseIntFV[] centroids = result2.centroids;
        HardAssigner<SparseIntFV, float[], IntFloatPair> assigner2 = result2.defaultHardAssigner();

        // Finding out the classifier accuracy
        for (ComputedImage trainingImage : trainingImages) {
            System.out.println("Classifying training... " + count);
            trainingImage.setCentroid(centroids[assigner2.assign(trainingImage.getExtractedFeature())]);
            count++;
        }
        count = 1;

        // Read testing images
        for (MBFImage testingImage : testing) {
            System.out.println("Reading testing... " + count);

            // Creates a BoVW for the image
            ComputedImage computedImage = computeImage(testingImage);
            BagOfVisualWords bagOfVisualWords = new BagOfVisualWords(assigner);
            computedImage.setExtractedFeature(bagOfVisualWords.aggregateVectorsRaw(extractFeatures(engine, computeImage(testingImage))));
            computedImage.setCentroid(centroids[assigner2.assign(computedImage.getExtractedFeature())]);
            testingImages.add(computedImage);
            count++;
        }
        count = 1;

        // Read testing images
        float correctClassificationCount = 0f;

        for (ComputedImage testingImage : testingImages) {
            System.out.println("Classifying testing... " + count);

            // Identify the subjects with the same centroid
            for (ComputedImage trainingImage : trainingImages) {
                if (trainingImage.getCentroid() == testingImage.getCentroid()) {
                    if (classificationTest(testingImage.getId(), trainingImage.getId())) {
                        DisplayUtilities.display(testingImage.getImage());
                        DisplayUtilities.display(trainingImage.getImage());
                        System.out.println("Classified " + testingImage.getId() + " to " + trainingImage.getImage());
                        correctClassificationCount += 1f;
                    }
                }
            }
            count++;
        }
        count = 1;

        System.out.println("Finished!");
        System.out.println("Classification accuracy = " + ((correctClassificationCount / 22f) * 100f) + "%");
    }

    static ComputedImage computeImage(MBFImage image) {
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