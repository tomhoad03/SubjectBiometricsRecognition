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

import java.io.IOException;
import java.nio.file.Paths;
import java.util.*;

public class Main {
    public static void main(String[] args) throws IOException {
        String path = Paths.get("").toAbsolutePath() + "\\src\\main\\java\\biometrics\\";
        VFSListDataset<MBFImage> training = new VFSListDataset<>(path + "training", ImageUtilities.MBFIMAGE_READER);
        VFSListDataset<MBFImage> testing = new VFSListDataset<>(path + "testing", ImageUtilities.MBFIMAGE_READER);

        ArrayList<ComputedImage> trainingImagesFront = new ArrayList<>();
        ArrayList<ComputedImage> trainingImagesSide = new ArrayList<>();
        ArrayList<ComputedImage> testingImagesFront = new ArrayList<>();
        ArrayList<ComputedImage> testingImagesSide = new ArrayList<>();

        // Read training images
        int count = 1;
        for (MBFImage trainingImage : training) {
            if (count % 2 == 1) {
                trainingImagesFront.add(readImage(trainingImage, count));
            } else {
                trainingImagesSide.add(readImage(trainingImage, count));
            }
            count++;
        }

        // Read the testing images
        count = 1;
        for (MBFImage testingImage : testing) {
            if (count % 2 == 0) {
                testingImagesFront.add(readImage(testingImage, count));
            } else {
                testingImagesSide.add(readImage(testingImage, count));
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

    static ComputedImage readImage(MBFImage image, int count) {
        // Resize and crop the image
        image = image.extractCenter(image.getWidth() / 2, (image.getHeight() / 2) + 120, 720, 1260);
        image.processInplace(new ResizeProcessor(0.5f));
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

        return new ComputedImage(count, personComponent, clonedImage);
    }

    // Classifies the dataset
    static double classifyImages(ArrayList<ComputedImage> trainingImages, ArrayList<ComputedImage> testingImages) {
        // Trains the assigner
        for (ComputedImage trainingImage : trainingImages) {
            trainingImage.setExtractedFeature(extractSilhouette(trainingImage).normaliseFV());
        }

        for (ComputedImage testingImage : testingImages) {
            testingImage.setExtractedFeature(extractSilhouette(testingImage).normaliseFV());
        }

        // K-Nearest Neighbours based on aspect ratio
        trainingImages.sort(Comparator.comparingDouble(ComputedImage::getAspectRatio));
        List<ComputedImage> kNearestTrainingImages = trainingImages.subList(0, 10);

        // Nearest neighbour to find the closest training image to each testing image
        float correctClassificationCount = 0f;

        for (ComputedImage testingImage : testingImages) {
            ComputedImage nearestImage = null;
            double nearestDistance = -1;

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
                DisplayUtilities.display(nearestImage.getImage());
                DisplayUtilities.display(testingImage.getImage());
                correctClassificationCount += 1f;
            }
        }
        return correctClassificationCount;
    }

    // Extract silhouette
    static DoubleFV extractSilhouette(ComputedImage image) {
        float[] array = image.getBoundaryDistances();

        ArrayList<Double> arrayList = new ArrayList<>();
        for (float value : array) {
            arrayList.add((double) value);
        }

        Random rand = new Random();
        int i = 0, maxSize = 1100;
        while (arrayList.size() > maxSize) {
            if (i == 64) {
                i = 0;
            }
            int subSize = arrayList.size() / 64;
            arrayList.remove(rand.nextInt(i * subSize, (i + 1) * subSize));
            i++;
        }

        double[] array2 = new double[maxSize];
        for (int j = 0; j < maxSize; j++) {
            array2[j] = arrayList.get(j);
        }

        return new DoubleFV(array2);
    }

    // Post classification test - not used in classification
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