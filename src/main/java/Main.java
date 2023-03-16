import gnu.trove.list.array.TFloatArrayList;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.colour.ColourSpace;
import org.openimaj.image.connectedcomponent.GreyscaleConnectedComponentLabeler;
import org.openimaj.image.pixel.ConnectedComponent;
import org.openimaj.image.processor.PixelProcessor;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.util.pair.IntFloatPair;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class Main {
    public static void main(String[] args) throws IOException {
        final VFSListDataset<MBFImage> training = new VFSListDataset<>(Paths.get("").toAbsolutePath() + "\\src\\main\\java\\biometrics\\training", ImageUtilities.MBFIMAGE_READER);
        final VFSListDataset<MBFImage> testing = new VFSListDataset<>(Paths.get("").toAbsolutePath() + "\\src\\main\\java\\biometrics\\testing", ImageUtilities.MBFIMAGE_READER);

        ArrayList<TrainingImage> featureVectors = new ArrayList<>();
        int globalCount = 0;

        for (MBFImage trainingImage : training) {
            trainingImage = ColourSpace.convert(trainingImage, ColourSpace.CIE_Lab);
            float[][] imageData = trainingImage.getPixelVectorNative(new float[trainingImage.getWidth() * trainingImage.getHeight()][3]);

            // Groups the pixels into their classes
            FloatKMeans cluster = FloatKMeans.createExact(2);
            FloatCentroidsResult result = cluster.cluster(imageData);
            float[][] centroids = result.centroids;

            // Assigns pixels to classes
            trainingImage.processInplace((PixelProcessor<Float[]>) pixel -> {
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
            List<ConnectedComponent> components = labeler.findComponents(trainingImage.flatten());
            TFloatArrayList componentFeatureVectors = new TFloatArrayList();

            // Append all component distance vectors
            for (ConnectedComponent component : components) {
                if (component.calculateArea() > 10000) {
                    componentFeatureVectors.addAll(component.calculateBoundaryDistanceFromCentre());
                }
            }
            featureVectors.add(new TrainingImage(trainingImage, componentFeatureVectors));

            System.out.println("Training... " + globalCount);
            globalCount++;
        }

        globalCount = 0;

        for (MBFImage testingImage : testing) {
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
            TFloatArrayList componentFeatureVectors = new TFloatArrayList();

            // Append all component distance vectors
            for (ConnectedComponent component : components) {
                if (component.calculateArea() > 10000) {
                    componentFeatureVectors.addAll(component.calculateBoundaryDistanceFromCentre());
                }
            }

            System.out.println("Testing... " + globalCount);

            float distance = -1;
            int count = 0;
            int closest = 0;

            for (TrainingImage trainingImage : featureVectors) {
                float[] array1 = trainingImage.getFeatureVector().toArray();
                float[] array2 = componentFeatureVectors.toArray();
                float sum = 0;

                for (int i = 0; i < array1.length; i++) {
                    sum += Math.exp(array1[i] - array2[i]);
                }
                float sqrt = (float) Math.sqrt(sum);

                if (distance == -1 || distance > sqrt) {
                    distance = sqrt;
                    closest = count;
                }
                count++;
            }

            System.out.println(testingImage + ", " + featureVectors.get(closest).getImage());

            testingImage = ColourSpace.convert(testingImage, ColourSpace.RGB);
            DisplayUtilities.display(testingImage);
        }
    }
}