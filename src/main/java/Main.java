import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.colour.ColourSpace;
import org.openimaj.image.connectedcomponent.GreyscaleConnectedComponentLabeler;
import org.openimaj.image.pixel.ConnectedComponent;
import org.openimaj.image.pixel.Pixel;
import org.openimaj.image.processor.PixelProcessor;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.util.pair.IntFloatPair;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.List;

public class Main {
    public static void main(String[] args) throws IOException {
        final VFSListDataset<MBFImage> training = new VFSListDataset<>(Paths.get("").toAbsolutePath() + "\\src\\main\\java\\biometrics\\training", ImageUtilities.MBFIMAGE_READER);
        final VFSListDataset<MBFImage> testing = new VFSListDataset<>(Paths.get("").toAbsolutePath() + "\\src\\main\\java\\biometrics\\testing", ImageUtilities.MBFIMAGE_READER);

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

            for (ConnectedComponent component : components) {
                if (component.calculateArea() > 100) {
                    Pixel centroidPixel = component.calculateCentroidPixel();
                    System.out.println(centroidPixel);
                }
            }

            trainingImage = ColourSpace.convert(trainingImage, ColourSpace.RGB);
            DisplayUtilities.display(trainingImage);
        }

    }
}