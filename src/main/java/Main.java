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
        final VFSListDataset<MBFImage> training = new VFSListDataset<>(Paths.get("").toAbsolutePath() + "\\src\\main\\java\\biometrics\\training", ImageUtilities.MBFIMAGE_READER);
        final VFSListDataset<MBFImage> testing = new VFSListDataset<>(Paths.get("").toAbsolutePath() + "\\src\\main\\java\\biometrics\\testing", ImageUtilities.MBFIMAGE_READER);

        int globalCount = 1;

        for (MBFImage trainingImage : training) {
            System.out.println("Training... " + globalCount);

            // Resize and crop the image
            trainingImage = trainingImage.extractCenter(trainingImage.getWidth() / 2, (trainingImage.getHeight() / 2) + 120, 720, 1260);
            MBFImage clonedTrainingImage = trainingImage.clone();

            // Apply a Guassian blur to reduce noise
            trainingImage = ColourSpace.convert(trainingImage, ColourSpace.CIE_Lab);
            trainingImage.processInplace(new FFastGaussianConvolve(2, 2));

            // Get the pixel data
            float[][] imageData = trainingImage.getPixelVectorNative(new float[trainingImage.getWidth() * trainingImage.getHeight()][3]);

            // Groups the pixels into their classes
            FloatKMeans cluster = FloatKMeans.createExact(2);
            FloatCentroidsResult result = cluster.cluster(imageData);
            float[][] centroids = result.centroids;

            // Assigns pixels to a class
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

            // Get the two connected components
            GreyscaleConnectedComponentLabeler labeler = new GreyscaleConnectedComponentLabeler();
            List<ConnectedComponent> components = labeler.findComponents(trainingImage.flatten());
            ArrayList<Float> componentFeatureVectors = new ArrayList<>();

            // Get the person component
            components.sort(Comparator.comparingInt(PixelSet::calculateArea));
            Collections.reverse(components);
            ConnectedComponent personComponent = components.get(1);

            // Get the boundary pixels and all contained pixels
            List<Pixel> boundary = personComponent.getOuterBoundary();
            Set<Pixel> pixels = personComponent.getPixels();

            // Remove all unnecessary pixels from image
            for (int y = 0; y < clonedTrainingImage.getHeight(); y++) {
                for (int x = 0; x < clonedTrainingImage.getWidth(); x++) {
                    if (!pixels.contains(new Pixel(x, y))) {
                        clonedTrainingImage.getBand(0).pixels[y][x] = 1;
                        clonedTrainingImage.getBand(1).pixels[y][x] = 1;
                        clonedTrainingImage.getBand(2).pixels[y][x] = 1;
                    }
                }
            }

            // Example display of image
            globalCount++;
            DisplayUtilities.display(clonedTrainingImage);
        }

        System.out.println("Finished!");
    }
}