import org.checkerframework.checker.units.qual.C;
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
import org.openimaj.image.segmentation.FelzenszwalbHuttenlocherSegmenter;
import org.openimaj.image.segmentation.SegmentationUtilities;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
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

        // Read testing images
        for (MBFImage testingImage : testing) {
            System.out.println("Reading testing... " + count);
            testingImages.add(computeImage(testingImage));
            count++;
        }

        count = 1;

        // Training the classifier
        for (ComputedImage trainingImage : trainingImages) {
            System.out.println("Training... " + count);
            DisplayUtilities.display(trainingImage.getDisplayImageB());
            count++;
        }

        System.out.println("Finished!");
    }

    static ComputedImage computeImage(MBFImage image) {
        // Resize and crop the image
        image = image.extractCenter(image.getWidth() / 2, (image.getHeight() / 2) + 120, 720, 1260);
        MBFImage clonedImageA = image.clone(), clonedImageB = image.clone();

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
        for (int y = 0; y < clonedImageA.getHeight(); y++) {
            for (int x = 0; x < clonedImageA.getWidth(); x++) {
                if (!pixels.contains(new Pixel(x, y))) {
                    clonedImageA.getBand(0).pixels[y][x] = 0;
                    clonedImageA.getBand(1).pixels[y][x] = 0;
                    clonedImageA.getBand(2).pixels[y][x] = 0;
                    clonedImageB.getBand(0).pixels[y][x] = 1;
                    clonedImageB.getBand(1).pixels[y][x] = 1;
                    clonedImageB.getBand(2).pixels[y][x] = 1;
                } else {
                    clonedImageA.getBand(0).pixels[y][x] = 1;
                    clonedImageA.getBand(1).pixels[y][x] = 1;
                    clonedImageA.getBand(2).pixels[y][x] = 1;
                }
            }
        }

        return new ComputedImage(count, true, personComponent, clonedImageA, clonedImageB);
    }
}