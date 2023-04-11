import ai.djl.inference.Predictor;
import ai.djl.modality.cv.BufferedImageFactory;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.Joints;
import ai.djl.translate.TranslateException;
import org.openimaj.feature.DoubleFV;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.colour.ColourSpace;
import org.openimaj.image.colour.RGBColour;
import org.openimaj.image.connectedcomponent.GreyscaleConnectedComponentLabeler;
import org.openimaj.image.pixel.ConnectedComponent;
import org.openimaj.image.pixel.Pixel;
import org.openimaj.image.pixel.PixelSet;
import org.openimaj.image.segmentation.KMSpatialColourSegmenter;
import org.openimaj.image.segmentation.SegmentationUtilities;
import org.openimaj.math.geometry.shape.Rectangle;

import javax.imageio.ImageIO;
import java.io.File;
import java.io.IOException;
import java.util.*;

public class PersonFV {
    private final int id;
    private final DoubleFV extractedFeature;

    /**
     * Creates a feature vector from an image
     * @param id The FV id
     * @param image The image
     * @param isTraining Is the image in the training or testing set?
     * @param PATH The relative file path
     * @param predictor The pose estimator predictor
     * @param temperatures The list of colours
     */
    public PersonFV(int id, MBFImage image, boolean isTraining, String PATH, Predictor<Image, Joints> predictor, Float[][] temperatures) throws IOException, TranslateException {
        this.id = id;

        // Crop the image
        image = image.extractCenter((image.getWidth() / 2) + 100, (image.getHeight() / 2) + 115, 740, 1280);
        MBFImage segmentedImage = image.clone();

        // Image segmentation
        KMSpatialColourSegmenter segmenter = new KMSpatialColourSegmenter(ColourSpace.CIE_Lab, 2);
        SegmentationUtilities.renderSegments(image, segmenter.segment(image));

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
        for (int y = 0; y < segmentedImage.getHeight(); y++) {
            for (int x = 0; x < segmentedImage.getWidth(); x++) {
                if (!pixels.contains(new Pixel(x, y))) {
                    segmentedImage.getBand(0).pixels[y][x] = 1;
                    segmentedImage.getBand(1).pixels[y][x] = 1;
                    segmentedImage.getBand(2).pixels[y][x] = 1;
                }
            }
        }

        // Print the original image
        String resultPath = isTraining ? "training" : "testing";
        File imageFile = new File(PATH + "segmented\\" + resultPath + "\\" + id + ".jpg");
        ImageUtilities.write(segmentedImage, imageFile);

        // Creates the new images
        MBFImage temperatureImage = segmentedImage.clone();
        MBFImage jointsImage = segmentedImage.clone();
        Rectangle boundingBox = component.calculateRegularBoundingBox();

        // Find the joints from the segmented image
        Image jointlessImage = BufferedImageFactory.getInstance().fromImage(ImageIO.read(imageFile));
        Joints joints = predictor.predict(jointlessImage);

        // Find the joints from the segmented image and draw them
        ArrayList<Pixel> jointPixels = new ArrayList<>();
        for (Joints.Joint joint : joints.getJoints()) {
            Pixel pixel = new Pixel((int) (joint.getX() * jointsImage.getWidth()), (int) (joint.getY() * jointsImage.getHeight()));
            jointsImage.drawPoint(pixel, RGBColour.RED, 6);
            jointPixels.add(pixel);
        }

        // Draw the component outline
        for (Pixel pixel : component.getOuterBoundary()) {
            jointsImage.drawPoint(pixel, RGBColour.GREEN, 4);
        }

        // Creates a pose model
        PoseModel poseModel = new PoseModel(jointPixels.get(0), // nose
                jointPixels.get(1), // right eye
                jointPixels.get(2), // left eye
                jointPixels.get(3), // right ear
                jointPixels.get(4), // left ear
                jointPixels.get(5), // right shoulder
                jointPixels.get(6), // left shoulder
                jointPixels.get(7), // right elbow
                jointPixels.get(8), // left elbow
                jointPixels.get(jointPixels.size() - 8), // right wrist*
                jointPixels.get(jointPixels.size() - 7), // left wrist*
                jointPixels.get(jointPixels.size() - 6), // right hip
                jointPixels.get(jointPixels.size() - 5), // left hip
                jointPixels.get(jointPixels.size() - 4), // right knee
                jointPixels.get(jointPixels.size() - 3), // left knee
                jointPixels.get(jointPixels.size() - 2), // right ankle
                jointPixels.get(jointPixels.size() - 1)); // left ankle

        // Draw the centroid point
        Pixel centroid = component.calculateCentroidPixel(); // calculateMiddle(calculateMiddle(poseModel.leftShoulder(), poseModel.rightShoulder()), calculateMiddle(poseModel.rightHip(), poseModel.leftHip()));
        jointsImage.drawPoint(centroid, RGBColour.BLUE, 6);

        // Print the joints image
        File jointsImageFile = new File(PATH + "joints\\" + resultPath + "\\" + id + ".jpg");
        ImageUtilities.write(jointsImage, jointsImageFile);

        // Creates the temperature image
        double[] temperatureCounts = new double[48];

        for (int y = 0; y < temperatureImage.getHeight(); y++) {
            for (int x = 0; x < temperatureImage.getWidth(); x++) {
                if (pixels.contains(new Pixel(x, y))) {
                    double divide = ((float) temperatureImage.getHeight() - (float) y) / boundingBox.getHeight();
                    double doubleIndex = (divide * temperatures.length) / 2f;
                    int index = (int) Math.floor(doubleIndex);

                    // Splits the image into left and right
                    if (x > centroid.getX()) {
                        index += temperatures.length / 2f;
                    }

                    // Sets the temperature of the pixel
                    try {
                        Float[] temperature = temperatures[index];
                        temperatureImage.getBand(0).pixels[y][x] = temperature[0];
                        temperatureImage.getBand(1).pixels[y][x] = temperature[1];
                        temperatureImage.getBand(2).pixels[y][x] = temperature[2];

                        if (temperatureCounts[index] == 0) {
                            temperatureCounts[index] = 1;
                        } else {
                            temperatureCounts[index] = temperatureCounts[index] + 1;
                        }
                    } catch (Exception ignored) { }
                }
            }
        }

        // Prints the temperature image
        File temperatureImageFile = new File(PATH + "temperature\\" + resultPath + "\\" + id + ".jpg");
        ImageUtilities.write(temperatureImage, temperatureImageFile);

        // Extract silhouette feature vector
        int maxBins = 56, halfBlankBinSize = 2, binCount = 0, backCount = 0;
        double[] doubleDistances = new double[maxBins - (halfBlankBinSize * 4)];
        ArrayList<PolarPixel> borderPixels = new ArrayList<>();

        for (Pixel pixel : component.getOuterBoundary()) {
            borderPixels.add(new PolarPixel(calculateDistance(pixel, centroid), calculateAngle(pixel, centroid)));
        }
        borderPixels.sort(Comparator.comparingDouble(PolarPixel::angle));

        ArrayList<Double> bin = new ArrayList<>();
        for (PolarPixel pixel : borderPixels) {
            bin.add(pixel.radius());

            if ((pixel.angle() > (((2 * Math.PI) / maxBins) * (binCount + 1))) || pixel == borderPixels.get(borderPixels.size() - 1)) {
                double sum = 0;
                for (double value : bin) {
                    sum += value;
                }

                if ((binCount > halfBlankBinSize && binCount < ((maxBins / 2) - halfBlankBinSize))
                        || (binCount > ((maxBins / 2) + halfBlankBinSize) && binCount < (maxBins - halfBlankBinSize))) {
                    doubleDistances[binCount - backCount] = sum / bin.size();
                } else {
                    backCount++;
                }
                binCount++;
                bin.clear();
            }
        }
        DoubleFV silhouetteFV = new DoubleFV(doubleDistances).normaliseFV();

        // Extract joints feature vector
        double[] jointsArray = new double[48];

        // Invariant features to centroid
        jointsArray[0] = calculateDistance(poseModel.nose(), centroid);
        jointsArray[1] = calculateDistance(poseModel.rightEye(), centroid);
        jointsArray[2] = calculateDistance(poseModel.leftEye(), centroid);
        jointsArray[3] = calculateDistance(poseModel.rightEar(), centroid);
        jointsArray[4] = calculateDistance(poseModel.leftEar(), centroid);
        jointsArray[5] = calculateDistance(poseModel.rightShoulder(), centroid);
        jointsArray[6] = calculateDistance(poseModel.leftShoulder(), centroid);
        jointsArray[7] = calculateDistance(poseModel.rightElbow(), centroid);
        jointsArray[8] = calculateDistance(poseModel.leftElbow(), centroid);
        jointsArray[9] = calculateDistance(poseModel.rightHip(), centroid);
        jointsArray[10] = calculateDistance(poseModel.leftHip(), centroid);
        jointsArray[11] = calculateDistance(poseModel.rightKnee(), centroid);
        jointsArray[12] = calculateDistance(poseModel.leftKnee(), centroid);
        jointsArray[13] = calculateDistance(poseModel.rightAnkle(), centroid);
        jointsArray[14] = calculateDistance(poseModel.leftAnkle(), centroid);

        // Inter-face distances
        jointsArray[15] = calculateDistance(poseModel.leftEar(), poseModel.leftEye());
        jointsArray[16] = calculateDistance(poseModel.leftEye(), poseModel.nose());
        jointsArray[17] = calculateDistance(poseModel.nose(), poseModel.rightEye());
        jointsArray[18] = calculateDistance(poseModel.rightEye(), poseModel.rightEar());
        jointsArray[19] = calculateDistance(poseModel.leftEar(), poseModel.rightEar());

        // Body widths
        jointsArray[20] = calculateDistance(poseModel.leftShoulder(), poseModel.rightShoulder());
        jointsArray[21] = calculateDistance(poseModel.leftElbow(), poseModel.rightElbow());
        jointsArray[22] = calculateDistance(poseModel.leftHip(), poseModel.rightHip());
        jointsArray[23] = calculateDistance(poseModel.leftKnee(), poseModel.rightKnee());
        jointsArray[24] = calculateDistance(poseModel.leftAnkle(), poseModel.rightAnkle());

        // Body heights
        jointsArray[25] = calculateDistance(poseModel.leftElbow(), poseModel.leftShoulder());
        jointsArray[26] = calculateDistance(poseModel.rightElbow(), poseModel.rightShoulder());
        jointsArray[27] = calculateDistance(poseModel.leftHip(), poseModel.leftShoulder());
        jointsArray[28] = calculateDistance(poseModel.rightHip(), poseModel.rightShoulder());
        jointsArray[29] = calculateDistance(poseModel.leftHip(), poseModel.leftElbow());
        jointsArray[30] = calculateDistance(poseModel.rightHip(), poseModel.rightElbow());
        jointsArray[31] = calculateDistance(poseModel.leftHip(), poseModel.leftKnee());
        jointsArray[32] = calculateDistance(poseModel.rightHip(), poseModel.rightKnee());
        jointsArray[33] = calculateDistance(poseModel.leftKnee(), poseModel.leftAnkle());
        jointsArray[34] = calculateDistance(poseModel.rightKnee(), poseModel.rightAnkle());

        for (int i = 35; i < 48; i++) {
            jointsArray[i] = 0;
        }
        DoubleFV jointsFV = new DoubleFV(jointsArray).normaliseFV();

        // Extract temperatures feature vector
        DoubleFV temperaturesFV = new DoubleFV(temperatureCounts).normaliseFV();

        this.extractedFeature = silhouetteFV.concatenate(jointsFV).concatenate(temperaturesFV);
    }

    /**
     * Calculate the distance between two pixels
     * @param pixelA Pixel A
     * @param pixelB Pixel B
     * @return The distance
     */
    public double calculateDistance(Pixel pixelA, Pixel pixelB) {
        return Math.sqrt(Math.pow(pixelA.getX() - pixelB.getX(), 2) + Math.pow(pixelA.getY() - pixelB.getY(), 2));
    }

    /**
     * Calculate the angle between two pixels
     * @param pixelA Pixel A
     * @param pixelB pixel B
     * @return The angle (radians)
     */
    public double calculateAngle(Pixel pixelA, Pixel pixelB) {
        double xDiff = pixelA.getX() - pixelB.getX(), yDiff = pixelA.getY() - pixelB.getY();
        double angle = Math.atan(yDiff / xDiff);

        if (xDiff < 0) {
            angle += Math.PI;
        } else if (xDiff > 0 && yDiff < 0) {
            angle += 2 * Math.PI;
        }
        if (xDiff == 0 && yDiff > 0) {
            angle = Math.PI / 2;
        } else if (xDiff == 0 && yDiff < 0) {
            angle = 3 * (Math.PI / 2);
        } else if (xDiff < 0 && yDiff == 0) {
            angle = Math.PI;
        } else if (xDiff > 0 && yDiff == 0) {
            angle = 0;
        }
        return angle;
    }

    /**
     * @return The FV id
     */
    public int getId() {
        return id;
    }

    /**
     * @return The FV
     */
    public DoubleFV getExtractedFeature() {
        return extractedFeature;
    }

    /**
     * A record to represent a pixel with polar coordinates
     * @param radius
     * @param angle
     */
    public record PolarPixel(double radius, double angle) { }

    /**
     * A record to represent a pose
     * @param nose
     * @param rightEye
     * @param leftEye
     * @param rightEar
     * @param leftEar
     * @param rightShoulder
     * @param leftShoulder
     * @param rightElbow
     * @param leftElbow
     * @param rightWrist
     * @param leftWrist
     * @param rightHip
     * @param leftHip
     * @param rightKnee
     * @param leftKnee
     * @param rightAnkle
     * @param leftAnkle
     */
    public record PoseModel(Pixel nose, Pixel rightEye, Pixel leftEye, Pixel rightEar, Pixel leftEar, Pixel rightShoulder,
                            Pixel leftShoulder, Pixel rightElbow, Pixel leftElbow, Pixel rightWrist, Pixel leftWrist,
                            Pixel rightHip, Pixel leftHip, Pixel rightKnee, Pixel leftKnee, Pixel rightAnkle,
                            Pixel leftAnkle) { }
}
