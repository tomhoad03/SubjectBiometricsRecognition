import gnu.trove.list.array.TFloatArrayList;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.colour.ColourSpace;
import org.openimaj.image.connectedcomponent.GreyscaleConnectedComponentLabeler;
import org.openimaj.image.pixel.ConnectedComponent;
import org.openimaj.image.pixel.Pixel;
import org.openimaj.image.pixel.PixelSet;
import org.openimaj.image.processing.convolution.FFastGaussianConvolve;
import org.openimaj.image.processing.edges.CannyEdgeDetector;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.image.processor.PixelProcessor;
import org.openimaj.image.typography.hershey.HersheyFont;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.util.pair.IntFloatPair;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Main {
    public static void main(String[] args) throws IOException {
        final VFSListDataset<MBFImage> training = new VFSListDataset<>(Paths.get("").toAbsolutePath() + "\\src\\main\\java\\biometrics\\training", ImageUtilities.MBFIMAGE_READER);
        final VFSListDataset<MBFImage> testing = new VFSListDataset<>(Paths.get("").toAbsolutePath() + "\\src\\main\\java\\biometrics\\testing", ImageUtilities.MBFIMAGE_READER);

        ArrayList<TrainingImage> featureVectors = new ArrayList<>();
        int globalCount = 1;

        for (MBFImage trainingImage : training) {
            System.out.println("Training... " + globalCount);

            trainingImage = trainingImage.extractCenter(trainingImage.getWidth() / 2, (trainingImage.getHeight() / 2) + 120, 720, 1260);
            trainingImage = ColourSpace.convert(trainingImage, ColourSpace.CIE_Lab);
            trainingImage.processInplace(new FFastGaussianConvolve(2, 2));

            float[][] imageData = trainingImage.getPixelVectorNative(new float[trainingImage.getWidth() * trainingImage.getHeight()][3]);

            // Groups the pixels into their classes
            FloatKMeans cluster = FloatKMeans.createExact(2);
            FloatCentroidsResult result = cluster.cluster(imageData);
            float[][] centroids = result.centroids;

            // Assigns pixels to classes
            trainingImage.processInplace((PixelProcessor<Float[]>) pixel -> {
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

            trainingImage = ColourSpace.convert(trainingImage, ColourSpace.RGB);
            trainingImage.processInplace(new CannyEdgeDetector());

            // Get the connected components
            GreyscaleConnectedComponentLabeler labeler = new GreyscaleConnectedComponentLabeler();
            List<ConnectedComponent> components = labeler.findComponents(trainingImage.flatten());
            ArrayList<Float> componentFeatureVectors = new ArrayList<>();

            // Append all component distance vectors
            for (ConnectedComponent component : components) {
                if (component.calculateArea() < 50) {
                    continue;
                }
                trainingImage.drawText("Point:", component.calculateCentroidPixel(), HersheyFont.TIMES_MEDIUM, 20);

                if (component.calculateArea() > 10000) {
                    TFloatArrayList boundaryDistances = component.calculateBoundaryDistanceFromCentre(); // TODO This size of array is to random and so vectors don't line up

                    float[] sortedBoundaryDistances = boundaryDistances.toArray();
                    float[] normalisedBoundaryDistances = boundaryDistances.toArray();
                    Arrays.sort(sortedBoundaryDistances);

                    for (float normalisedBoundaryDistance : normalisedBoundaryDistances) {
                        componentFeatureVectors.add(normalisedBoundaryDistance / sortedBoundaryDistances[sortedBoundaryDistances.length - 1]);
                    }
                }
            }

            globalCount++;
            DisplayUtilities.display(trainingImage);
        }

        globalCount = 1;

        for (MBFImage testingImage : testing) {
            System.out.println("Testing... " + globalCount);

            testingImage = ColourSpace.convert(testingImage, ColourSpace.CIE_Lab);
            float[][] imageData = testingImage.getPixelVectorNative(new float[testingImage.getWidth() * testingImage.getHeight()][3]);

            // Groups the pixels into their classes
            FloatKMeans cluster = FloatKMeans.createExact(2);
            FloatCentroidsResult result = cluster.cluster(imageData);
            float[][] centroids = result.centroids;

            // Assigns pixels to classes
            testingImage.processInplace((PixelProcessor<Float[]>) pixel -> {
                HardAssigner<float[], float[], IntFloatPair> assigner = result.defaultHardAssigner();

                float[] set1 = new float[3];
                for (int i = 0; i < 3; i++){
                    set1[i] = pixel[i];
                }
                float[] centroid = centroids[assigner.assign(set1)];

                Float[] set2 = new Float[3];
                for (int i = 0; i < 3; i++){
                    set2[i] = centroid[i];
                }
                return set2;
            });

            // Get the connected components
            GreyscaleConnectedComponentLabeler labeler = new GreyscaleConnectedComponentLabeler();
            List<ConnectedComponent> components = labeler.findComponents(testingImage.flatten());
            ArrayList<Float> componentFeatureVectors = new ArrayList<>();

            // Append all component distance vectors
            for (ConnectedComponent component : components) {
                if (component.calculateArea() > 10000) {
                    TFloatArrayList boundaryDistances = component.calculateBoundaryDistanceFromCentre(); // TODO This size of array is to random and so vectors don't line up

                    float[] sortedBoundaryDistances = boundaryDistances.toArray();
                    float[] normalisedBoundaryDistances = boundaryDistances.toArray();
                    Arrays.sort(sortedBoundaryDistances);

                    for (float normalisedBoundaryDistance : normalisedBoundaryDistances) {
                        componentFeatureVectors.add(normalisedBoundaryDistance / sortedBoundaryDistances[sortedBoundaryDistances.length - 1]);
                    }
                }
            }

            float distance = -1;
            int count = 0;
            int closest = 0;

            for (TrainingImage trainingImage : featureVectors) {
                ArrayList<Float> trainingFeatureVectors = trainingImage.getFeatureVector();
                double sum = 0f;

                for (int i = 0; i < trainingFeatureVectors.size() + componentFeatureVectors.size(); i++) {
                    float value1 = 0f, value2 = 0f;

                    if (i < trainingFeatureVectors.size()) {
                        value1 = trainingFeatureVectors.get(i);
                    }
                    if (i < componentFeatureVectors.size()) {
                        value2 = componentFeatureVectors.get(i);
                    }

                    if (value1 > 0f || value2 > 0f) {
                        sum += Math.exp(value1 - value2);
                    } else {
                        break;
                    }
                }
                float sqrt = (float) Math.sqrt(sum);

                if (distance == -1 || distance > sqrt) {
                    distance = sqrt;
                    closest = count;
                }
                count++;
            }

            System.out.println(globalCount + " - " + featureVectors.get(closest).getId() + " - " + distance);
            globalCount++;

            testingImage = ColourSpace.convert(testingImage, ColourSpace.RGB);
            DisplayUtilities.display(testingImage);
        }

        System.out.println("Finished!");
    }
}